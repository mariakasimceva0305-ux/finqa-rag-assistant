"""
Сервис для работы с эмбеддингами
"""

import numpy as np
import pickle
import time
from typing import List, Tuple, Optional
from openai import OpenAI
from tqdm import tqdm
from .config import config


class EmbeddingService:
    """Сервис для создания и управления эмбеддингами"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or config.EMBEDDER_API_KEY
        self.base_url = base_url or config.EMBEDDING_API_URL
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Инициализация клиента OpenAI"""
        if not self.api_key:
            raise ValueError("API ключ для эмбеддингов не установлен")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def get_embedding(self, text: str, model: str = None, max_retries: int = 3) -> List[float]:
        """Получение эмбеддинга для текста"""
        if model is None:
            model = config.EMBEDDING_MODEL
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=model,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Ошибка при получении эмбеддинга (попытка {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2)
                else:
                    print(f"Не удалось получить эмбеддинг после {max_retries} попыток: {e}")
                    raise
        
        return []
    
    def create_embeddings_batch(self, texts: List[str], model: str = None) -> np.ndarray:
        """Создание эмбеддингов для батча текстов"""
        embeddings = []
        
        for text in tqdm(texts, desc="Создание эмбеддингов"):
            embedding = self.get_embedding(text, model)
            embeddings.append(embedding)
            time.sleep(0.1)  # Задержка для избежания rate limit
        
        return np.array(embeddings)
    
    def load_or_create_embeddings(self, articles_data: List[Tuple], 
                                  cache_embeddings: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Загрузка эмбеддингов из кэша или создание новых"""
        
        # Извлекаем тексты чанков
        chunks = [chunk_text for _, chunk_text, _ in articles_data]
        
        # Проверяем кэш
        if cache_embeddings:
            try:
                with open(config.EMBEDDINGS_CACHE_PATH, 'rb') as f:
                    embeddings = pickle.load(f)
                with open(config.CHUNKS_CACHE_PATH, 'rb') as f:
                    cached_chunks = pickle.load(f)
                
                if len(embeddings) == len(chunks):
                    print(f"Загружено {len(embeddings)} эмбеддингов из кэша")
                    return embeddings, chunks
            except FileNotFoundError:
                print("Кэш не найден, создаем новые эмбеддинги...")
        
        # Создаем новые эмбеддинги
        print("Создание новых эмбеддингов...")
        embeddings = self.create_embeddings_batch(chunks)
        
        # Сохраняем в кэш
        if cache_embeddings:
            with open(config.EMBEDDINGS_CACHE_PATH, 'wb') as f:
                pickle.dump(embeddings, f)
            with open(config.CHUNKS_CACHE_PATH, 'wb') as f:
                pickle.dump(chunks, f)
            print(f"Сохранено {len(embeddings)} эмбеддингов в кэш")
        
        return embeddings, chunks
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Вычисление косинусной близости между векторами"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
