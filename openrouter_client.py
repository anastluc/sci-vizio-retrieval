"""
Legacy wrapper for backwards compatibility.
"""
from sci_vizio_retrieval.client import OpenRouterVision, VisionResponse
from sci_vizio_retrieval.config import VISION_MODEL as DEFAULT_MODEL

__all__ = ["OpenRouterVision", "VisionResponse", "DEFAULT_MODEL"]
