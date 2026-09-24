"""Central configuration loaded from environment / .env file."""

from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="V2G_", env_file=".env", env_file_encoding="utf-8")

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    llm_max_tokens: int = 32768  # default output cap per chat call; call sites with a fixed budget pass explicit values
    llm_cache: bool = True  # content-addressed raw-response cache (V2G_LLM_CACHE=0 disables)
    media_cache: bool = True  # shared download/transcode cache (V2G_MEDIA_CACHE=0 → artifacts go to the run's work dir)

    # Godot
    godot_path: str = "godot"

    # Video extraction
    max_duration: int = (
        120  # seconds — cap for a single analysis; detail-mode input beyond this is trimmed
    )
    frame_interval: float = 2.0  # seconds between extracted frames (short videos)
    video_max_mb: int = 20  # max video size (MB) for direct upload to LLM
    frame_budget: int = 40  # max keyframes to send to LLM (long video cap)
    scene_threshold: float = 0.3  # ffmpeg scene-detect sensitivity (0.0–1.0, lower = more frames)
    chunk_duration: int = 60  # seconds per analysis chunk (detail mode); must be ≤ max_duration

    # Image generation (local Qwen-Image-2.1 via the optional `imagegen` extra)
    imagegen_provider: str = ""  # "" = off (assets stay raw frames) | "qwen" = local Qwen-Image-2.1
    imagegen_model: str = (
        ""  # model root override (diffusers dir / HF id); "" = managed GGUF bundle
    )
    imagegen_style: str = ""  # global style prefix for all image gen prompts
    imagegen_steps: int = 40  # denoising steps (Qwen's default)
    imagegen_max_side: int = 1024  # longest output edge — lower to save RAM/VRAM
    imagegen_autorestyle: bool = False  # redraw assets even without -i; prompt = design.style

    # Output
    output_root: Path = Path("projects")

    @model_validator(mode="after")
    def _chunk_fits_single_analysis(self) -> "Settings":
        """A chunk is one analysis unit — it must fit inside the per-analysis cap."""
        if self.chunk_duration > self.max_duration:
            raise ValueError(
                f"V2G_CHUNK_DURATION={self.chunk_duration} exceeds "
                f"V2G_MAX_DURATION={self.max_duration}: chunk length must not "
                "be greater than the single-analysis cap"
            )
        return self


settings = Settings()
