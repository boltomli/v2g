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
    max_duration: int = 120  # seconds
    frame_interval: float = 2.0  # seconds between extracted frames
    video_max_mb: int = 20  # max video size (MB) for direct upload to LLM

    # Output
    output_root: Path = Path("projects")


settings = Settings()
