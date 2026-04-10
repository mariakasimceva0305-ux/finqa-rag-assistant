"""
Модуль для поиска релевантных чанков и реранкинга
"""

import numpy as np
import requests
import time
from typing import List, Tuple
from .config import config
from .embedding_service import EmbeddingService


class Retriever:
    """Поиск и реранкинг релевантных чанков"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
    
    def search_relevant_chunks(self, query: str, embeddings: np.ndarray, 
                               chunks: List[str], top_k: int = None) -> List[Tuple[str, float]]:
        """Поиск наиболее релевантных чанков"""
        if top_k is None:
            top_k = config.TOP_K
        
        # Получаем эмбеддинг запроса
        query_embedding = np.array(self.embedding_service.get_embedding(query))
        
        # Вычисляем косинусную близость
        similarities = []
        for i, chunk_embedding in enumerate(embeddings):
            similarity = self.embedding_service.cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((i, similarity))
        
        # Сортируем по убыванию сходства
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Возвращаем топ-K результатов
        top_results = similarities[:top_k]
        return [(chunks[idx], score) for idx, score in top_results]
    
    def rerank_chunks(self, query: str, chunk_items: List[Tuple[str, float]], 
                      api_key: str = None) -> List[Tuple[str, float]]:
        """Реранкинг результатов через API"""
        if not chunk_items:
            return chunk_items
        
        api_key = api_key or config.EMBEDDER_API_KEY
        
        try:
            documents = [text for text, _ in chunk_items]
            
            # Подготовка запроса к API реранкера
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": config.RERANK_MODEL,
                "query": query,
                "documents": documents
            }
            
            # Отправка запроса
            response = requests.post(
                config.RERANK_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Обработка разных форматов ответа
                if isinstance(data, dict) and "results" in data:
                    results = data["results"]
                    # Сортируем по score
                    sorted_indices = sorted(range(len(results)), 
                                           key=lambda i: results[i].get("score", 0), 
                                           reverse=True)
                    
                    # Переупорядочиваем чанки
                    reranked = [chunk_items[i] for i in sorted_indices]
                    return reranked
            
            # В случае ошибки API возвращаем исходный порядок
            print(f"Ошибка реранкинга: {response.status_code}")
            return chunk_items
            
        except Exception as e:
            print(f"Ошибка при реранкинге: {e}")
            return chunk_items
    
    def retrieve(self, query: str, embeddings: np.ndarray, chunks: List[str], 
                 use_rerank: bool = True) -> List[str]:
        """Полный процесс поиска и реранкинга"""
        # Поиск релевантных чанков
        relevant_chunks = self.search_relevant_chunks(query, embeddings, chunks)
        
        # Реранкинг если включен
        if use_rerank and relevant_chunks:
            relevant_chunks = self.rerank_chunks(query, relevant_chunks)
        
        # Возвращаем только тексты чанков
        return [chunk for chunk, _ in relevant_chunks]
