from abc import ABC, abstractmethod

class VisionAPI(ABC):
    @abstractmethod
    def analyze_image(self, image_base64: str, prompt: str) -> str:
        pass