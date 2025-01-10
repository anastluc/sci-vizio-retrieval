
from vision_models.claude_vision import ClaudeVision
from vision_models.gemini_vision import GeminiVision
from vision_models.groq_vision import GroqVision
from vision_models.openai_vision import OpenAIVision
from vision_models.vision_API import VisionAPI
from vision_models.xai_vision import XAI_Vision


def create_vision_api(specific_model:str) -> VisionAPI:
    if specific_model == "openai":
        return OpenAIVision(specific_model)
    elif specific_model == "anthropic":
        return ClaudeVision(specific_model)
    elif specific_model in ["gemini-2.0-flash-exp"]:
        return GeminiVision(specific_model)
    elif specific_model == "groq-vision":
        return GroqVision(specific_model)
    elif specific_model == "xai":
        return XAI_Vision(specific_model)
    else:
        raise ValueError(f"Unsupported provider: {specific_model}")