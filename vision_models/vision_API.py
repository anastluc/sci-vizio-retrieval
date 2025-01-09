from abc import ABC, abstractmethod

class VisionAPI(ABC):
    @abstractmethod
    def analyze_image(self, image_base64: str, prompt: str) -> str:
        pass

def create_vision_api(self,specific_model:str) -> VisionAPI:
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