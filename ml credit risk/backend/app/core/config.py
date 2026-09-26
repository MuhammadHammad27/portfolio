import os
from pathlib import Path
from typing import List, Union

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field

    class Settings(BaseSettings):
        PROJECT_NAME: str = "Explainable Credit Risk Assessment API"
        VERSION: str = "1.0.0"
        API_V1_STR: str = "/api/v1"
        ENVIRONMENT: str = "development"
        HOST: str = "0.0.0.0"
        PORT: int = 8000
        
        # Paths
        BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
        MODEL_PATH: str = os.getenv(
            "MODEL_PATH",
            str(BASE_DIR / "model" / "loan_model.pkl")
        )
        
        # CORS
        CORS_ORIGINS: List[str] = [
            "http://localhost:3000",
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "*"
        ]
        
        class Config:
            env_file = ".env"
            case_sensitive = True

    settings = Settings()

except ImportError:
    # Lightweight fallback if pydantic_settings is not installed
    class Settings:
        PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Explainable Credit Risk Assessment API")
        VERSION: str = "1.0.0"
        API_V1_STR: str = "/api/v1"
        ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
        HOST: str = os.getenv("HOST", "0.0.0.0")
        PORT: int = int(os.getenv("PORT", 8000))
        
        BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
        MODEL_PATH: str = os.getenv(
            "MODEL_PATH",
            str(BASE_DIR / "model" / "loan_model.pkl")
        )
        
        CORS_ORIGINS: List[str] = [
            "http://localhost:3000",
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "*"
        ]

    settings = Settings()
