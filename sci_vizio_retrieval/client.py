import base64
import logging
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from sci_vizio_retrieval.config import (
    VISION_API_DELAY,
    VISION_MAX_RETRIES,
    VISION_MAX_TOKENS,
    VISION_MODEL,
)

logger = logging.getLogger(__name__)

@dataclass
class VisionResponse:
    """Minimal wrapper so callers can access .text consistently."""
    text: str

class OpenRouterVision:
    """
    Vision LLM client that uses OpenRouter as a unified gateway.

    Args:
        model: OpenRouter model identifier. If None, resolves from config.ini/VISION_MODEL.
        api_delay: Seconds to wait between API calls. If None, resolves from config.ini/VISION_API_DELAY.
        max_retries: Number of retry attempts. If None, resolves from config.ini/VISION_MAX_RETRIES.
        max_tokens: Response token limit. If None, resolves from config.ini/VISION_MAX_TOKENS.
        client: Pre-built OpenAI-compatible client (mainly for tests). If None,
            one is created for OpenRouter using OPENROUTER_API_KEY.
    """

    def __init__(
        self,
        model: str = None,
        api_delay: float = None,
        max_retries: int = None,
        max_tokens: int = None,
        client=None,
    ):
        if client is None:
            load_dotenv()
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY not found. "
                    "Set it in your .env file or as an environment variable."
                )
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

        self.client = client
        self.model = model if model is not None else VISION_MODEL
        self.api_delay = api_delay if api_delay is not None else VISION_API_DELAY
        self.max_retries = max(1, max_retries if max_retries is not None else VISION_MAX_RETRIES)
        self.max_tokens = max_tokens if max_tokens is not None else VISION_MAX_TOKENS

    @staticmethod
    def _mime_type(image_path: str | Path) -> str:
        mime, _ = mimetypes.guess_type(str(image_path))
        return mime if mime and mime.startswith("image/") else "image/jpeg"

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
        with open(image_path, "rb") as f:
            data = f.read()
        return self.analyze_image_bytes(data, self._mime_type(image_path), prompt)

    def analyze_image_bytes(self, data: bytes, mime_type: str, prompt: str) -> VisionResponse:
        """Like :meth:`analyze_image`, for an image already in memory."""
        image_b64 = base64.b64encode(data).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
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
                    max_tokens=self.max_tokens,
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
