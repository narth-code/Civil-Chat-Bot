from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    app_name: str = "Civil Reference Router"
    log_dir: str = "logs"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_user: str = "civil_chatbot_owner"
    postgres_password: str = ""
    postgres_db: str = "civil_chatbot"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    backend_host: str = "127.0.0.1"
    backend_port: int = Field(default=8000, ge=1, le=65535)
    tavily_api_key: str = ""
    web_search_domains: list[str] = ["library.municode.com"]
    web_search_max_results: int = Field(default=5, ge=1)
    max_retries: int = Field(default=3, ge=0)
    reviewer_model: str = "qwen2.5:7b"
    retrieval_top_k: int = Field(default=8, ge=1)
    retrieval_fts_weight: float = Field(default=0.0, ge=0.0)
    retrieval_embedding_variant: Literal["", "ctx1", "ctx2"] = "ctx1"
    retrieval_rerank: bool = True
    retrieval_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    retrieval_rerank_top_n: int = Field(default=30, ge=1)
    retrieval_relevance_threshold: float = (
        1.0  # top reranker logit below this = nothing relevant in the KB (calibrated)
    )
    retrieval_grader_margin: float = Field(default=2.0, gt=0)  # scores this far below threshold still go to the grader
    ollama_num_ctx: int = Field(default=8192, ge=2048)  # qwen3 context window; covers the longest measured prompt
    ollama_keep_alive: int = -1  # seconds Ollama keeps a model resident; -1 = until the backend stops
    ollama_timeout_seconds: float = Field(default=120.0, gt=0)  # bound on each model request, in seconds
    reviewer_num_gpu: int | None = 0  # reviewer layers on GPU; 0 = CPU, None = Ollama default
    reranker_batch_size: int = Field(default=8, ge=1)  # passages per cross-encoder forward pass
    retrieval_expand_abbreviations: bool = False
    corpus_city: str = "Austin"

    @property
    def database_url(self) -> URL:
        return URL.create(
            "postgresql+asyncpg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
