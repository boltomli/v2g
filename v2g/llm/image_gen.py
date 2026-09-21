"""Image generation/transformation provider interface.

Currently a no-op (NullProvider) — assets come directly from video frames.
Reserves the interface for future image-generation APIs (DALL-E, Stable Diffusion, etc.)
that can transform extracted assets to match a different style (--instruct).

Future usage:
    provider = get_provider()          # returns configured provider
    transformed = provider.transform(
        image_path=Path("assets/char_hero.png"),
        prompt="restyle as medieval knight, pixel art",
        reference="character: knight in shining armor",
    )

Configuration (env vars / .env):
    V2G_IMAGEGEN_API_KEY     — API key for image gen service
    V2G_IMAGEGEN_BASE_URL    — endpoint (default: https://api.openai.com/v1)
    V2G_IMAGEGEN_MODEL       — model name (default: dall-e-3)
    V2G_IMAGEGEN_STYLE       — global style prefix prepended to all prompts
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class ImageGenProvider(abc.ABC):
    """Abstract interface for image generation/transformation."""

    @abc.abstractmethod
    def transform(
        self,
        image_path: Path,
        prompt: str,
        *,
        reference: str = "",
        size: str = "1024x1024",
    ) -> Path:
        """Transform an image according to *prompt*.

        Args:
            image_path: Source image to transform.
            prompt: Style/content transformation instruction.
            reference: Additional context (e.g. character description from design).
            size: Output dimensions.

        Returns:
            Path to the transformed image (may be the same as input if no-op).
        """
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider is configured and ready."""
        ...


class NullProvider(ImageGenProvider):
    """No-op provider — returns the original image unchanged.

    Used when no image gen API is configured (the default).
    """

    def transform(self, image_path: Path, prompt: str, **_: object) -> Path:
        return image_path

    def is_available(self) -> bool:
        return True  # always "available" (just does nothing)


# ── Future providers (stubs) ─────────────────────────────────────────────────
# Uncomment and implement when the API is ready.

# class OpenAIImageProvider(ImageGenProvider):
#     """DALL-E / GPT-image based transformation."""
#
#     def __init__(self, api_key: str, base_url: str, model: str, style_prefix: str):
#         self._api_key = api_key
#         self._base_url = base_url
#         self._model = model
#         self._style_prefix = style_prefix
#
#     def transform(self, image_path: Path, prompt: str, *, reference: str = "", size: str = "1024x1024") -> Path:
#         import base64
#         from openai import OpenAI
#         client = OpenAI(api_key=self._api_key, base_url=self._base_url)
#         full_prompt = f"{self._style_prefix} {prompt}".strip()
#         if reference:
#             full_prompt += f"\nReference: {reference}"
#
#         # Read source image
#         img_b64 = base64.b64encode(image_path.read_bytes()).decode()
#         # Use edits or generations endpoint depending on model
#         response = client.images.edit(
#             model=self._model,
#             image=image_path,
#             prompt=full_prompt,
#             size=size,
#         )
#         # Save result
#         import httpx
#         out_path = image_path.parent / f"{image_path.stem}_gen.png"
#         img_data = httpx.get(response.data[0].url).content
#         out_path.write_bytes(img_data)
#         return out_path
#
#     def is_available(self) -> bool:
#         return bool(self._api_key)


# class StableDiffusionProvider(ImageGenProvider):
#     """Stable Diffusion / ComfyUI / local API transformation."""
#     ...


# ── Provider factory ─────────────────────────────────────────────────────────


def get_provider() -> ImageGenProvider:
    """Return the configured image generation provider.

    Currently always returns NullProvider. When V2G_IMAGEGEN_API_KEY is set,
    this will return the appropriate API provider.
    """
    # Future: check config and return API provider
    # from v2g.config import settings
    # if settings.imagegen_api_key:
    #     return OpenAIImageProvider(
    #         api_key=settings.imagegen_api_key,
    #         base_url=settings.imagegen_base_url,
    #         model=settings.imagegen_model,
    #         style_prefix=settings.imagegen_style,
    #     )
    return NullProvider()
