from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str = "dummy_key_for_testing"
    groq_model: str = "llama3-groq-70b-8192-tool-use"
    database_url: str = "postgresql+asyncpg://megaai:megaai@db:5432/megaai"
    sync_database_url: str = "postgresql://megaai:megaai@db:5432/megaai"
    redis_url: str = "redis://redis:6379/0"
    secret_key: str = "dev_secret"
    eval_concurrency: int = 2
    max_tool_retries: int = 2
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
