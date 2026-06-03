"""High-level pipelines — the user-facing glue."""

from ._base import PipelineInfo
from ._anima import anima_calibrate_oss
from .image_to_image import ImageToImage
from .inpaint import Inpaint
from .text_to_image import TextToImage

__all__ = ["ImageToImage", "Inpaint", "PipelineInfo", "TextToImage", "anima_calibrate_oss"]
