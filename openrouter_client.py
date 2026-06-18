"""
OpenRouter Vision API client.

A single client that routes to any vision-capable LLM via the OpenRouter API
(https://openrouter.ai). Uses the OpenAI-compatible chat completions endpoint.
"""

import base64
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

# Default model — can be any OpenRouter vision-capable model slug.
# Browse available models at https://openrouter.ai/models
DEFAULT_MODEL = "google/gemini-2.0-flash-001"


@dataclass
class VisionResponse:
    """Minimal wrapper so callers can access .text consistently."""
    text: str


class OpenRouterVision:
    """
    Vision LLM client that uses OpenRouter as a unified gateway.

    Args:
        model: OpenRouter model identifier (e.g. "google/gemini-2.0-flash-001",
               "openai/gpt-4o", "anthropic/claude-sonnet-4").
        api_delay: Seconds to wait between API calls to avoid rate limits.
        max_retries: Number of retry attempts on transient failures.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_delay: float = 2.0,
        max_retries: int = 3,
    ):
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. "
                "Set it in your .env file or as an environment variable."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model
        self.api_delay = api_delay
        self.max_retries = max_retries

    @staticmethod
    def _encode_image(image_path: str | Path) -> str:
        """Read and base64-encode a local image file."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_image(self, image_path: str | Path, prompt: str) -> VisionResponse:
        """
        Send an image to the vision model and return its text response.

        Args:
            image_path: Path to a local image file (jpg, png, etc.)
            prompt: Text prompt describing what analysis to perform.

        Returns:
            VisionResponse with a .text attribute containing the model output.

        Raises:
            Exception: After all retries are exhausted.
        """
        image_b64 = self._encode_image(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            }
        ]

        last_exception = None
        for attempt in range(self.max_retries):
            if self.api_delay > 0:
                time.sleep(self.api_delay)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0.7,
                )
                content = response.choices[0].message.content
                return VisionResponse(text=content)

            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = 2 ** (attempt + 1)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)

        raise last_exception
