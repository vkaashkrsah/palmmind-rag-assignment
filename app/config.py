from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "sqlite+aiosqlite:///./data/app.db"
    chat_history_limit: int = 10
    chat_ttl_seconds: int = 86400
    top_k: int = 4
    max_upload_bytes: int = 10 * 1024 * 1024


settings = Settings()
