import base64
from vision_models.vision_API import VisionAPI
import os
import google.generativeai as genai
from dotenv import load_dotenv
import time

class GeminiVision(VisionAPI):
    def __init__(self, model: str):
        load_dotenv()
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        self.API_TIME_DELAY = 5
    
        
    def analyze_image(self, image_path: str, prompt: str) -> str:
        max_retries: int = 3
        initial_delay: float = 1.0

        print(f"Delaying {self.API_TIME_DELAY} seconds to avoid API's throttling")
        time.sleep(self.API_TIME_DELAY)

        # Create the model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name=self.model,
            generation_config=generation_config,
        )

        response = None
        last_exception = None

        for attempt in range(max_retries):
            print(f"Attempt {attempt + 1} of {max_retries}")
            try:
                files = [self.upload_to_gemini(image_path)]
                # print(files[0])
                
                
                chat_session = model.start_chat(
                    history=[
                        {
                            "role": "user",
                            "parts": [files[0]],
                        },
                    ]
                )

                response = chat_session.send_message(prompt)
                # If we get here without an exception, we have a successful response
                return response

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:  # Not the last attempt
                    delay = initial_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
                    time.sleep(delay)
                continue

        # If we get here, all attempts failed
        if last_exception:
            print(f"Error occurred after {max_retries} attempts: {str(last_exception)}")
            raise last_exception

        return response

    def upload_to_gemini(self, path):
        """Uploads the given file to Gemini.

        See https://ai.google.dev/gemini-api/docs/prompting_with_media
        """
        file = genai.upload_file(path, mime_type="image/png")
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file