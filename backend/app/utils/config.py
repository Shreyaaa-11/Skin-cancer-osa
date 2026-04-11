from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Skin Cancer Classifier API"
    model_path: Path = Path("backend/artifacts/model.keras")
    thresholds_path: Path = Path("backend/artifacts/thresholds.json")
    enable_tta: bool = True
    cors_origins: str = "*"


settings = Settings()
