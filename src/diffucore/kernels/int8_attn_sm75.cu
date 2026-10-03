// INT8-QK / FP16-accumulated-PV flash attention for sm75 (Turing), head_dim 128.
//
// GeForce Turing runs fp16 MMA with fp32 accumulation at half rate, which is what
// FlashAttention-2 uses for both GEMMs. Measured on an RTX 2060: fp32-acc HMMA
// 24 TFLOPS, fp16-acc HMMA 49, INT8 IMMA 98. So:
//   * S = Q K^T runs on INT8 IMMA. Q is quantized per token and K per 64-key block,
//     after subtracting K's per-channel mean (softmax ignores per-row constants).
//     softmax_scale * log2(e) is folded into the Q scales.
//   * P V runs on fp16-accumulate HMMA. The fp16 partials hold at most
//     FLUSH_EVERY * 64 keys before being added into fp32 accumulators.
//   * The row max is taken on the int32 scores, and int32 -> float is a magic-number
//     add folded into the exp2 argument (no I2F per score).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int D = 128;
constexpr int BR = 128;   // query rows per CTA (8 warps x 16)
constexpr int BC = 64;    // keys per tile
constexpr int NW = 8;
constexpr int NT = NW * 32;
// fp16 partials sum at most FLUSH_EVERY * 64 terms p * v with p <= 1. Anima's V
// reaches |v| ~ 136, so 2 keeps the worst case (128 * 136) far below 65504.
constexpr int FLUSH_EVERY = 2;
constexpr float LOG2E = 1.4426950408889634f;
constexpr int MAGIC_I = 0x4B400000;   // bits of 1.5 * 2^23
constexpr float MAGIC_F = 12582912.f;

// ------------------------------------------------------------------ prep

// Per-(b,h,d) sum of K over tokens; block = 256 threads = 16 rows x 16 chunks.
__global__ void k_sum_kernel(const half* __restrict__ k, float* __restrict__ ksum, int L, int H,
                             int64_t sb, int64_t sl, int64_t sh, int rows_per_block) {
  const int b = blockIdx.z, h = blockIdx.y;
  const int ch = threadIdx.x & 15, rg = threadIdx.x >> 4;
  const half* base = k + b * sb + h * sh + ch * 8;
  float acc[8] = {};
  const int r0 = blockIdx.x * rows_per_block;
  const int r1 = min(L, r0 + rows_per_block);
  for (int r = r0 + rg; r < r1; r += 16) {
    uint4 u = *reinterpret_cast<const uint4*>(base + r * sl);
    const half2* hv = reinterpret_cast<const half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 f = __half22float2(hv[i]);
      acc[2 * i] += f.x;
      acc[2 * i + 1] += f.y;
    }
  }
  __shared__ float red[16][129];
#pragma unroll
  for (int i = 0; i < 8; ++i) red[rg][ch * 8 + i] = acc[i];
  __syncthreads();
  if (threadIdx.x < D) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) s += red[i][threadIdx.x];
    atomicAdd(ksum + (b * H + h) * D + threadIdx.x, s);
  }
}

// K - mean, int8 per 64-token block. One CTA per (tile, h, b); 256 threads x 4 chunks of 8.
__global__ void k_quant_kernel(const half* __restrict__ k, const float* __restrict__ ksum,
                               int8_t* __restrict__ ki, float* __restrict__ ks, int L, int H,
                               int Lpad, int64_t sb, int64_t sl, int64_t sh) {
  const int b = blockIdx.z, h = blockIdx.y, tile = blockIdx.x;
  const int bh = b * H + h;
  const float inv_l = 1.f / L;
  float vals[4][8];
  float amax = 0.f;
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const int ci = threadIdx.x + u * NT;
    const int r = ci >> 4, c = ci & 15;
    const int row = tile * BC + r;
    if (row < L) {
      uint4 raw = *reinterpret_cast<const uint4*>(k + b * sb + row * sl + h * sh + c * 8);
      const half2* hv = reinterpret_cast<const half2*>(&raw);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 f = __half22float2(hv[i]);
        vals[u][2 * i] = f.x - ksum[bh * D + c * 8 + 2 * i] * inv_l;
        vals[u][2 * i + 1] = f.y - ksum[bh * D + c * 8 + 2 * i + 1] * inv_l;
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) vals[u][i] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(vals[u][i]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
  __shared__ float wmax[NW];
  if ((threadIdx.x & 31) == 0) wmax[threadIdx.x >> 5] = amax;
  __syncthreads();
  amax = wmax[0];
#pragma unroll
  for (int i = 1; i < NW; ++i) amax = fmaxf(amax, wmax[i]);
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  if (threadIdx.x == 0) ks[bh * (Lpad / BC) + tile] = amax / 127.f;
#pragma unroll
  for (int u = 0; u < 4; ++u) {
    const int ci = threadIdx.x + u * NT;
    const int r = ci >> 4, c = ci & 15;
    char4 q0, q1;
    q0.x = (int8_t)__float2int_rn(vals[u][0] * inv); q0.y = (int8_t)__float2int_rn(vals[u][1] * inv);
    q0.z = (int8_t)__float2int_rn(vals[u][2] * inv); q0.w = (int8_t)__float2int_rn(vals[u][3] * inv);
    q1.x = (int8_t)__float2int_rn(vals[u][4] * inv); q1.y = (int8_t)__float2int_rn(vals[u][5] * inv);
    q1.z = (int8_t)__float2int_rn(vals[u][6] * inv); q1.w = (int8_t)__float2int_rn(vals[u][7] * inv);
    int8_t* dst = ki + ((int64_t)bh * Lpad + tile * BC + r) * D + c * 8;
    reinterpret_cast<char4*>(dst)[0] = q0;
    reinterpret_cast<char4*>(dst)[1] = q1;
  }
}

// Q per-token int8; the stored scale also carries softmax_scale * log2(e).
__global__ void q_quant_kernel(const half* __restrict__ q, int8_t* __restrict__ qi, float* __restrict__ qs,
                               int L, int H, int Lpad, int64_t sb, int64_t sl, int64_t sh, float scale_log2,
                               int64_t ntok) {
  const int64_t w = (int64_t)blockIdx.x * NW + (threadIdx.x >> 5);
  if (w >= ntok) return;
  const int lane = threadIdx.x & 31;
  const int l = w % Lpad;
  const int64_t bh = w / Lpad;
  const int b = bh / H, h = bh % H;
  float f[4] = {0.f, 0.f, 0.f, 0.f};
  if (l < L) {
    uint2 raw = *reinterpret_cast<const uint2*>(q + b * sb + l * sl + h * sh + lane * 4);
    const half2* hv = reinterpret_cast<const half2*>(&raw);
    float2 a = __half22float2(hv[0]), c = __half22float2(hv[1]);
    f[0] = a.x; f[1] = a.y; f[2] = c.x; f[3] = c.y;
  }
  float amax = fmaxf(fmaxf(fabsf(f[0]), fabsf(f[1])), fmaxf(fabsf(f[2]), fabsf(f[3])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
  const float inv = amax > 0.f ? 127.f / amax : 0.f;
  char4 r;
  r.x = (int8_t)__float2int_rn(f[0] * inv); r.y = (int8_t)__float2int_rn(f[1] * inv);
  r.z = (int8_t)__float2int_rn(f[2] * inv); r.w = (int8_t)__float2int_rn(f[3] * inv);
  reinterpret_cast<char4*>(qi + w * D)[lane] = r;
  if (lane == 0) qs[w] = amax / 127.f * scale_log2;
}

// ------------------------------------------------------------------ attention

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
__device__ __forceinline__ void imma(int& c0, int& c1, uint32_t a, uint32_t b) {
  asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n"
      : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
}
__device__ __forceinline__ void hmma(uint32_t& c0, uint32_t& c1, uint32_t a0, uint32_t a1, uint32_t b) {
  asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
      : "+r"(c0), "+r"(c1) : "r"(a0), "r"(a1), "r"(b));
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
  __half2 h = __floats2half2_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}

struct Params {
  const int8_t* qi; const int8_t* ki; const half* v; half* o;
  const float* qs; const float* ks;
  int Lq, Lk, Lq_pad, Lk_pad, H;
  int64_t v_sb, v_sl, v_sh, o_sb, o_sl, o_sh;
};

template <bool TAIL>
__device__ __forceinline__ void tile_step(const Params& p, int tile, int lane, uint32_t sK, uint32_t sV,
                                          const uint32_t (&qf)[2][8], const float (&cq)[2], float ksc,
                                          float (&m)[2], float (&lsum)[2], float (&o)[16][4],
                                          uint32_t (&acc)[16][2], float (&beta)[2], bool flush) {
  const int t4 = lane & 3, rr = lane & 7, mi = lane >> 3;
  int s[2][8][2];
#pragma unroll
  for (int rg = 0; rg < 2; ++rg)
#pragma unroll
    for (int j = 0; j < 8; ++j) s[rg][j][0] = s[rg][j][1] = 0;

#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    uint32_t kf[8];
#pragma unroll
    for (int pq = 0; pq < 2; ++pq) {
      const int j = pq * 4 + mi;
      uint32_t r[4];
      ldsm_x4(r, sK + (j * 8 + rr) * 128 + ((kk ^ rr) << 4));
#pragma unroll
      for (int i = 0; i < 4; ++i) kf[pq * 4 + i] = r[i];
    }
#pragma unroll
    for (int rg = 0; rg < 2; ++rg)
#pragma unroll
      for (int j = 0; j < 8; ++j) imma(s[rg][j][0], s[rg][j][1], qf[rg][kk], kf[j]);
  }

  if (TAIL) {
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int i = 0; i < 2; ++i)
        if (tile * BC + j * 8 + 2 * t4 + i >= p.Lk) s[0][j][i] = s[1][j][i] = INT_MIN;
  }

  // Lazy max: the cross-thread reduction and the rescale only run when some score exceeds the
  // running max (exact: otherwise the new max equals the old one and alpha == 1).
  float c[2], bias[2];
  int lmax[2];
  bool up = false;
#pragma unroll
  for (int rg = 0; rg < 2; ++rg) {
    int mx = s[rg][0][0];
#pragma unroll
    for (int j = 0; j < 8; ++j) mx = max(mx, max(s[rg][j][0], s[rg][j][1]));
    lmax[rg] = mx;
    c[rg] = cq[rg] * ksc;
    up |= (float)mx * c[rg] > m[rg];
  }
  if (__any_sync(0xffffffff, up)) {
    // New max: rescale the fp16 partials in place and defer the fp32 rescale to the next flush.
    float alpha[2];
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      int mx = lmax[rg];
      mx = max(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      mx = max(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      const float mnew = fmaxf(m[rg], (float)mx * c[rg]);
      alpha[rg] = ex2(m[rg] - mnew);
      m[rg] = mnew;
      lsum[rg] *= alpha[rg];
      beta[rg] *= alpha[rg];
    }
    const __half2 a0 = __float2half2_rn(alpha[0]), a1 = __float2half2_rn(alpha[1]);
#pragma unroll
    for (int nd = 0; nd < 16; ++nd) {
      *reinterpret_cast<__half2*>(&acc[nd][0]) = __hmul2(*reinterpret_cast<__half2*>(&acc[nd][0]), a0);
      *reinterpret_cast<__half2*>(&acc[nd][1]) = __hmul2(*reinterpret_cast<__half2*>(&acc[nd][1]), a1);
    }
  }
#pragma unroll
  for (int rg = 0; rg < 2; ++rg) bias[rg] = fmaf(MAGIC_F, c[rg], m[rg]);

  // j-outer PV into fp16 accumulators that persist for FLUSH_EVERY tiles before being flushed to fp32.
  float ls[2] = {0.f, 0.f};
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    uint32_t pf[2];
#pragma unroll
    for (int rg = 0; rg < 2; ++rg) {
      float e0 = ex2(fmaf(__int_as_float(s[rg][j][0] + MAGIC_I), c[rg], -bias[rg]));
      float e1 = ex2(fmaf(__int_as_float(s[rg][j][1] + MAGIC_I), c[rg], -bias[rg]));
      if (TAIL) {
        if (tile * BC + j * 8 + 2 * t4 >= p.Lk) e0 = 0.f;
        if (tile * BC + j * 8 + 2 * t4 + 1 >= p.Lk) e1 = 0.f;
      }
      ls[rg] += e0 + e1;
      pf[rg] = pack_h2(e0, e1);
    }
#pragma unroll
    for (int pq = 0; pq < 4; ++pq) {
      uint32_t r[4];
      ldsm_x4_t(r, sV + (j * 8 + rr) * 256 + (((pq * 4 + mi) ^ rr) << 4));
#pragma unroll
      for (int i = 0; i < 4; ++i) hmma(acc[pq * 4 + i][0], acc[pq * 4 + i][1], pf[0], pf[1], r[i]);
    }
  }
  lsum[0] += ls[0];
  lsum[1] += ls[1];
  if (flush) {
#pragma unroll
    for (int nd = 0; nd < 16; ++nd) {
      const float2 lo = __half22float2(*reinterpret_cast<__half2*>(&acc[nd][0]));
      const float2 hi = __half22float2(*reinterpret_cast<__half2*>(&acc[nd][1]));
      o[nd][0] = fmaf(o[nd][0], beta[0], lo.x);
      o[nd][1] = fmaf(o[nd][1], beta[0], lo.y);
      o[nd][2] = fmaf(o[nd][2], beta[1], hi.x);
      o[nd][3] = fmaf(o[nd][3], beta[1], hi.y);
      acc[nd][0] = acc[nd][1] = 0;
    }
    beta[0] = beta[1] = 1.f;
  }
}

__global__ void __launch_bounds__(NT, 1) attn_kernel(const Params p) {
  __shared__ __align__(128) uint8_t smem[BC * D + BC * D * 2];   // K int8 (8 KB) | V fp16 (16 KB)
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int qt = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int bh = b * p.H + h;
  const uint32_t sbase = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t sK = sbase, sV = sbase + BC * D;

  // Q tile (128 x 128 int8 = 16 KB) staged through the same buffer.
  {
    const int8_t* src = p.qi + ((int64_t)bh * p.Lq_pad + qt * BR) * D;
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int ci = tid + u * NT, r = ci >> 3, c = ci & 7;
      uint4 val = reinterpret_cast<const uint4*>(src)[ci];
      *reinterpret_cast<uint4*>(smem + r * 128 + ((c ^ (r & 7)) << 4)) = val;
    }
  }
  __syncthreads();
  uint32_t qf[2][8];
  {
    const int rr = lane & 7, mi = lane >> 3;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int f = i * 4 + mi, rg = f >> 3, kk = f & 7;
      const int row = warp * 16 + rg * 8 + rr;
      uint32_t r[4];
      ldsm_x4(r, sbase + row * 128 + ((kk ^ rr) << 4));
#pragma unroll
      for (int m2 = 0; m2 < 4; ++m2) {
        const int f2 = i * 4 + m2;
        qf[f2 >> 3][f2 & 7] = r[m2];
      }
    }
  }
  float cq[2];
  {
    const int g = lane >> 2;
    const float* qsrc = p.qs + (int64_t)bh * p.Lq_pad + qt * BR + warp * 16;
    cq[0] = qsrc[g];
    cq[1] = qsrc[g + 8];
  }
  __syncthreads();

  const int ntiles = (p.Lk + BC - 1) / BC;
  const int8_t* kbase = p.ki + (int64_t)bh * p.Lk_pad * D;
  const half* vbase = p.v + b * p.v_sb + h * p.v_sh;
  uint4 kreg[2], vreg[4];
  auto gload = [&](int tile) {
    const int8_t* ksrc = kbase + (int64_t)tile * BC * D;
#pragma unroll
    for (int u = 0; u < 2; ++u) kreg[u] = reinterpret_cast<const uint4*>(ksrc)[tid + u * NT];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int ci = tid + u * NT, r = ci >> 4, c = ci & 15;
      const int row = tile * BC + r;
      vreg[u] = row < p.Lk ? *reinterpret_cast<const uint4*>(vbase + row * p.v_sl + c * 8) : make_uint4(0, 0, 0, 0);
    }
  };
  auto sstore = [&]() {
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int ci = tid + u * NT, r = ci >> 3, c = ci & 7;
      *reinterpret_cast<uint4*>(smem + r * 128 + ((c ^ (r & 7)) << 4)) = kreg[u];
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int ci = tid + u * NT, r = ci >> 4, c = ci & 15;
      *reinterpret_cast<uint4*>(smem + BC * D + r * 256 + ((c ^ (r & 7)) << 4)) = vreg[u];
    }
  };

  gload(0);
  sstore();
  __syncthreads();

  float m[2] = {-INFINITY, -INFINITY}, lsum[2] = {0.f, 0.f};
  float o[16][4];
#pragma unroll
  for (int nd = 0; nd < 16; ++nd) o[nd][0] = o[nd][1] = o[nd][2] = o[nd][3] = 0.f;
  uint32_t acc[16][2];
#pragma unroll
  for (int nd = 0; nd < 16; ++nd) acc[nd][0] = acc[nd][1] = 0;
  float beta[2] = {1.f, 1.f};

  const float* ksp = p.ks + (int64_t)bh * (p.Lk_pad / BC);
  const bool has_tail = (p.Lk % BC) != 0;
  for (int tile = 0; tile < ntiles; ++tile) {
    const bool more = tile + 1 < ntiles;
    if (more) gload(tile + 1);
    const float ksc = ksp[tile];
    const bool flush = !more || (tile % FLUSH_EVERY) == FLUSH_EVERY - 1;
    if (has_tail && !more)
      tile_step<true>(p, tile, lane, sK, sV, qf, cq, ksc, m, lsum, o, acc, beta, flush);
    else
      tile_step<false>(p, tile, lane, sK, sV, qf, cq, ksc, m, lsum, o, acc, beta, flush);
    __syncthreads();
    if (more) {
      sstore();
      __syncthreads();
    }
  }

#pragma unroll
  for (int rg = 0; rg < 2; ++rg) {
    lsum[rg] += __shfl_xor_sync(0xffffffff, lsum[rg], 1);
    lsum[rg] += __shfl_xor_sync(0xffffffff, lsum[rg], 2);
  }
  const int g = lane >> 2, t4 = lane & 3;
  const float inv0 = 1.f / lsum[0], inv1 = 1.f / lsum[1];
  const int row0 = qt * BR + warp * 16 + g, row1 = row0 + 8;
  half* obase = p.o + b * p.o_sb + h * p.o_sh + 2 * t4;
#pragma unroll
  for (int nd = 0; nd < 16; ++nd) {
    if (row0 < p.Lq)
      *reinterpret_cast<uint32_t*>(obase + row0 * p.o_sl + nd * 8) = pack_h2(o[nd][0] * inv0, o[nd][1] * inv0);
    if (row1 < p.Lq)
      *reinterpret_cast<uint32_t*>(obase + row1 * p.o_sl + nd * 8) = pack_h2(o[nd][2] * inv1, o[nd][3] * inv1);
  }
}

void check_input(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kHalf, name, " must be fp16 CUDA");
  TORCH_CHECK(t.dim() == 4 && t.size(3) == D && t.stride(3) == 1, name, " must be (B, L, H, 128) with contiguous D");
  TORCH_CHECK(t.stride(2) % 8 == 0 && t.stride(1) % 8 == 0 && t.stride(0) % 8 == 0, name, " strides must be multiples of 8");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

}  // namespace

// q, k, v: (B, L, H, 128) fp16, any (B, L, H) strides. Returns (B, Lq, H, 128) contiguous.
torch::Tensor fwd(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale) {
  check_input(q, "q"); check_input(k, "k"); check_input(v, "v");
  const at::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int B = q.size(0), Lq = q.size(1), H = q.size(2), Lk = k.size(1);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && k.size(2) == H && v.size(2) == H && v.size(1) == Lk,
              "q, k, v shapes disagree");
  TORCH_CHECK(B > 0 && H > 0 && Lq > 0 && Lk > 0, "empty attention input");
  const int Lq_pad = (Lq + BR - 1) / BR * BR, Lk_pad = (Lk + BC - 1) / BC * BC;
  auto i8 = q.options().dtype(torch::kInt8);
  auto f32 = q.options().dtype(torch::kFloat32);
  auto qi = torch::empty({B, H, Lq_pad, D}, i8);
  auto ki = torch::empty({B, H, Lk_pad, D}, i8);
  auto qs = torch::empty({B, H, Lq_pad}, f32);
  auto ks = torch::empty({B, H, Lk_pad / BC}, f32);
  auto ksum = torch::zeros({B, H, D}, f32);
  auto o = torch::empty({B, Lq, H, D}, q.options());

  const int rpb = 256;
  k_sum_kernel<<<dim3((Lk + rpb - 1) / rpb, H, B), 256, 0, stream>>>(
      (const half*)k.data_ptr(), ksum.data_ptr<float>(), Lk, H, k.stride(0), k.stride(1), k.stride(2), rpb);
  k_quant_kernel<<<dim3(Lk_pad / BC, H, B), NT, 0, stream>>>(
      (const half*)k.data_ptr(), ksum.data_ptr<float>(), ki.data_ptr<int8_t>(), ks.data_ptr<float>(), Lk, H, Lk_pad,
      k.stride(0), k.stride(1), k.stride(2));
  const int64_t ntok = (int64_t)B * H * Lq_pad;
  q_quant_kernel<<<(ntok + NW - 1) / NW, NT, 0, stream>>>(
      (const half*)q.data_ptr(), qi.data_ptr<int8_t>(), qs.data_ptr<float>(), Lq, H, Lq_pad, q.stride(0), q.stride(1),
      q.stride(2), (float)sm_scale * LOG2E, ntok);

  Params p;
  p.qi = qi.data_ptr<int8_t>(); p.ki = ki.data_ptr<int8_t>();
  p.v = (const half*)v.data_ptr(); p.o = (half*)o.data_ptr();
  p.qs = qs.data_ptr<float>(); p.ks = ks.data_ptr<float>();
  p.Lq = Lq; p.Lk = Lk; p.Lq_pad = Lq_pad; p.Lk_pad = Lk_pad; p.H = H;
  p.v_sb = v.stride(0); p.v_sl = v.stride(1); p.v_sh = v.stride(2);
  p.o_sb = o.stride(0); p.o_sl = o.stride(1); p.o_sh = o.stride(2);
  attn_kernel<<<dim3(Lq_pad / BR, H, B), NT, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &fwd, "INT8-QK / FP16-PV attention (sm75, head_dim 128)");
}
