"""High-level pipelines — the user-facing glue."""

from ._base import PipelineInfo
from .image_to_image import ImageToImage
from .inpaint import Inpaint
from .text_to_image import TextToImage

__all__ = ["ImageToImage", "Inpaint", "PipelineInfo", "TextToImage"]
