from vision_models.vision_API import VisionAPI
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import time
class OpenAIVision(VisionAPI):
    def __init__(self, model: str):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.API_TIME_DELAY = 1

    def encode_image(self, card_image):
        with open(card_image, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        return base64_image
    
    def analyze_image(self, image_path: str, prompt: str) -> str:

        print(f"Delaying {self.API_TIME_DELAY} seconds to avoid API's throttling")
        time.sleep(self.API_TIME_DELAY)

        image_base64 = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": f"{prompt}"
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                    }
                ]
                }
            ],
            response_format={
                "type": "text"
            },
            temperature=1,
            max_completion_tokens=50,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
        return response.choices[0].message.content
