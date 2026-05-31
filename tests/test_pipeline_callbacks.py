from diffucore import PipelineInfo
from diffucore.pipelines._base import _step_progress


def test_step_progress_forwards_callback():
    seen = []
    with _step_progress(3, lambda step, total: seen.append((step, total))) as on_step:
        for i in range(3):
            on_step(i)

    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_pipeline_info_reports_decode_mode():
    info = PipelineInfo(vae_decode_mode="tiled")

    assert info.vae_decode_mode == "tiled"
