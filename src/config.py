"""
Конфигурация системы
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Конфигурация RAG системы"""
    # API ключи (загружаются из переменных окружения)
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    EMBEDDER_API_KEY: str = os.getenv("EMBEDDER_API_KEY", "")
    
    # Базовые URL API
    EMBEDDING_API_URL: str = "https://ai-for-finance-hack.up.railway.app/"
    RERANK_API_URL: str = "https://ai-for-finance-hack.up.railway.app/rerank"
    
    # Параметры обработки текста
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 3
    
    # Модели по умолчанию
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "openrouter/meta-llama/llama-3-70b-instruct"
    RERANK_MODEL: str = "deepinfra/Qwen/Qwen3-Reranker-4B"
    
    # Пути к данным (файлы в корне репозитория)
    TRAIN_DATA_PATH: str = "./train_data.csv"
    QUESTIONS_PATH: str = "./questions.csv"
    OUTPUT_PATH: str = "./data/processed/submission.csv"
    
    # Параметры генерации
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 1000
    MAX_CONTEXT_LENGTH: int = 4000
    
    # Кэширование
    EMBEDDINGS_CACHE_PATH: str = "./embeddings_cache.pkl"
    CHUNKS_CACHE_PATH: str = "./chunks_cache.pkl"
    
    def validate(self):
        """Проверка конфигурации"""
        if not self.LLM_API_KEY or not self.EMBEDDER_API_KEY:
            raise ValueError("API ключи не установлены. Проверьте файл .env")
        return self


config = Config()
