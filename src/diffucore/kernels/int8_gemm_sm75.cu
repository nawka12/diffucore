// SPDX-FileCopyrightText: Copyright (c) 2026 Dzakwan Haq (KayfaHaarukku, github.com/nawka12)
// SPDX-License-Identifier: Apache-2.0
//
// INT8 x INT8 -> fp16 GEMM for Turing (sm75) with the W8A8 rescale fused into
// the epilogue: out[m, n] = fp16(float(acc[m, n]) * (sx[m] * sw[n])). The same
// rounding steps as the int32 GEMM + Triton rescale pass it replaces, without
// writing and re-reading the int32 accumulator. Built on CUTLASS 2.x (headers
// from the nvidia-cutlass wheel): its sm75 i8816 IMMA mainloop, the family
// cuBLASLt runs, plus an epilogue visitor tree.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// The visitor headers name cutlass::gemm types, so the GEMM headers go first.
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

using namespace cute;
namespace tb = cutlass::epilogue::threadblock;
constexpr auto kRound = cutlass::FloatRoundStyle::round_to_nearest;

template <int kTileM>
struct ScaledGemm {
  using TileShape = cutlass::gemm::GemmShape<kTileM, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using OutMap = tb::OutputTileThreadLayout<TileShape, WarpShape, cutlass::half_t, 8, 1>;

  // fp16(acc * (sx * sw))
  using SX = tb::VisitorColBroadcast<OutMap, float, Stride<_1, _0, int32_t>>;
  using SW = tb::VisitorRowBroadcast<OutMap, float, Stride<_0, _1, int32_t>>;
  using Scale = tb::Sm80EVT<tb::VisitorCompute<cutlass::multiplies, float, float, kRound>, SX, SW>;
  using Y = tb::Sm80EVT<tb::VisitorCompute<cutlass::multiplies, cutlass::half_t, float, kRound>,
                        tb::VisitorAccFetch, Scale>;
  using Store = tb::VisitorAuxStore<OutMap, cutlass::half_t, kRound, Stride<int64_t, _1, int64_t>>;
  using Out = tb::Sm80EVT<Store, Y>;

  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
      int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
      cutlass::half_t, cutlass::layout::RowMajor, 8,
      int32_t, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
      TileShape, WarpShape, cutlass::gemm::GemmShape<8, 8, 16>, Out,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 2,
      cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static void run(const torch::Tensor& a, const torch::Tensor& w, const torch::Tensor& sx,
                  const torch::Tensor& sw, torch::Tensor& out) {
    const int M = a.size(0), K = a.size(1), N = w.size(0);
    typename Y::Arguments y{
        {},
        {{sx.data_ptr<float>(), 0.f, {_1{}, _0{}, int32_t(M)}},
         {sw.data_ptr<float>(), 0.f, {_0{}, _1{}, int32_t(N)}},
         {}},
        {}};
    typename Store::Arguments store{reinterpret_cast<cutlass::half_t*>(out.data_ptr()),
                                    {int64_t(N), _1{}, int64_t(M) * N}};
    typename Out::Arguments epilogue{y, store};
    typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, epilogue,
                                  a.data_ptr(), w.data_ptr(), nullptr, nullptr,
                                  int64_t(M) * K, int64_t(N) * K, 0, 0,
                                  int64_t(K), int64_t(K), int64_t(N), int64_t(N));
    Gemm gemm;
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "int8 gemm: shape not supported");
    TORCH_CHECK(gemm.initialize(args, nullptr, stream) == cutlass::Status::kSuccess, "int8 gemm: init failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "int8 gemm: launch failed");
  }
};

// a [M, K] int8, w [N, K] int8 (nn.Linear layout), sx [M] / sw [N] fp32.
torch::Tensor scaled_mm(torch::Tensor a, torch::Tensor w, torch::Tensor sx, torch::Tensor sw) {
  TORCH_CHECK(a.is_cuda() && a.dtype() == torch::kInt8 && w.dtype() == torch::kInt8, "int8 gemm: int8 CUDA inputs");
  TORCH_CHECK(a.dim() == 2 && w.dim() == 2 && a.size(1) == w.size(1), "int8 gemm: shapes");
  TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && sx.is_contiguous() && sw.is_contiguous(),
              "int8 gemm: contiguous inputs");
  TORCH_CHECK(sx.dtype() == torch::kFloat32 && sw.dtype() == torch::kFloat32, "int8 gemm: fp32 scales");
  TORCH_CHECK(a.size(1) % 16 == 0 && w.size(0) % 8 == 0, "int8 gemm: K % 16 and N % 8");
  const c10::cuda::CUDAGuard guard(a.device());
  auto out = torch::empty({a.size(0), w.size(0)}, a.options().dtype(torch::kHalf));
  // 256-row tiles win from ~1k rows up; small M (text-context projections) wastes them.
  if (a.size(0) >= 256) ScaledGemm<256>::run(a, w, sx, sw, out);
  else ScaledGemm<128>::run(a, w, sx, sw, out);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scaled_mm", &scaled_mm, "int8 GEMM with fused per-row x per-column rescale");
}
