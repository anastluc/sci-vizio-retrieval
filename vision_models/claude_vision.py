import anthropic
from vision_models.vision_API import VisionAPI
from dotenv import load_dotenv
import os
import base64 
import time

class ClaudeVision(VisionAPI):
    def __init__(self, model: str):
        self.model = model
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Client(api_key=api_key)


        

    def encode_image(self, card_image):
        with open(card_image, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        return base64_image
    
    def analyze_image(self, image_path: str, prompt: str) -> str:


        image_base64 = self.encode_image(image_path)
        message = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}<image>"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": "</image>"
                        }
                    ]
                }
            ]
        )
        return message.content[0].text
