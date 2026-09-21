"""Central configuration loaded from environment / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="V2G_", env_file=".env", env_file_encoding="utf-8")

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"

    # Godot
    godot_path: str = "godot"

    # Video extraction
    max_duration: int = 120  # seconds — max single-upload duration for detail mode
    frame_interval: float = 2.0  # seconds between extracted frames (short videos)
    video_max_mb: int = 20  # max video size (MB) for direct upload to LLM
    frame_budget: int = 40  # max keyframes to send to LLM (long video cap)
    scene_threshold: float = 0.3  # ffmpeg scene-detect sensitivity (0.0–1.0, lower = more frames)
    chunk_duration: int = 600  # seconds per analysis chunk for long videos (detail mode)

    # Image generation (reserved for future use)
    imagegen_api_key: str = ""
    imagegen_base_url: str = "https://api.openai.com/v1"
    imagegen_model: str = "dall-e-3"
    imagegen_style: str = ""  # global style prefix for all image gen prompts

    # Output
    output_root: Path = Path("projects")


settings = Settings()
