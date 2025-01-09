import requests
from vision_models.vision_API import VisionAPI
import os
from dotenv import load_dotenv
from typing import Literal
import base64
import time

class GroqVision(VisionAPI):
    VALID_MODELS = Literal["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]
    
    def __init__(
        self,
        model: VALID_MODELS = "llama-3.2-90b-vision-preview"
    ):        
        load_dotenv()
        GROQ_API_URL = os.getenv("GROQ_API_URL")
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.api_key = GROQ_API_KEY
        self.api_endpoint = GROQ_API_URL
        
        if model not in ["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]:
            raise ValueError(
                f"Invalid model: {model}. Must be one of: llama-3.2-11b-vision-preview, llama-3.2-90b-vision-preview"
            )
        self.model = model
        self.API_TIME_DELAY = 7


    def encode_image(self, card_image):
        with open(card_image, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        return base64_image
    
    def analyze_image(self, image_path: str, prompt: str) -> str:

        print(f"Delaying {self.API_TIME_DELAY} seconds to avoid API's throttling")
        time.sleep(self.API_TIME_DELAY)
        image_base64 = self.encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                    {"type": "text", "text": f"{prompt}"}            
                ]
            }
        ]

        # Make API request
        response = self.make_api_request(messages)

        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}")
        
        return response.json()["choices"][0]["message"]["content"]

    def make_api_request(self, messages: list) -> requests.Response:
        """Make request to Groq API."""
        return requests.post(
            self.api_endpoint,
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 1000
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=30
        )