import os
from pydantic import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN")
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    SIMILARITY_THRESHOLD: float = 0.1
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MONGO_URI: str = os.getenv("MONGO_URI")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME")

    class Config:
        env_file = ".env"

settings = Settings()