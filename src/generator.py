"""
Генератор ответов с использованием LLM
"""

from openai import OpenAI
from typing import List
from .config import config


class AnswerGenerator:
    """Генератор ответов на основе найденного контекста"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or config.LLM_API_KEY
        self.base_url = base_url or config.EMBEDDING_API_URL
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Инициализация клиента OpenAI"""
        if not self.api_key:
            raise ValueError("API ключ для LLM не установлен")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def create_prompt(self, question: str, context_chunks: List[str]) -> str:
        """Создание промпта для LLM"""
        
        # Объединяем контекстные чанки
        if not context_chunks:
            context_text = "Релевантный контекст не найден в базе знаний."
        else:
            context_parts = []
            for i, chunk in enumerate(context_chunks, 1):
                context_parts.append(f"[Контекст {i}]\n{chunk}")
            context_text = "\n\n".join(context_parts)
        
        # Обрезаем контекст если слишком длинный
        if len(context_text) > config.MAX_CONTEXT_LENGTH:
            context_text = context_text[:config.MAX_CONTEXT_LENGTH] + "...\n[Контекст обрезан]"
        
        # Создаем промпт
        prompt = f"""Ты финансовый консультант. Ответь на вопрос пользователя, используя предоставленный контекст из статей по финансам.

Если ответ не содержится в контексте, ответь на основе своих знаний, но укажи это.

Контекст:
{context_text}

Вопрос: {question}

Ответь подробно и точно на русском языке:"""
        
        return prompt
    
    def generate_answer(self, question: str, context_chunks: List[str], 
                        model: str = None) -> str:
        """Генерация ответа с использованием LLM"""
        if model is None:
            model = config.LLM_MODEL
        
        try:
            # Создаем промпт
            prompt = self.create_prompt(question, context_chunks)
            
            # Генерация ответа
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Ошибка при генерации ответа: {e}")
            
            # Fallback: простой ответ без контекста
            try:
                fallback_prompt = f"Ответь на вопрос на русском языке: {question}"
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": [{"type": "text", "text": fallback_prompt}]}],
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_TOKENS // 2
                )
                return response.choices[0].message.content
            except:
                return f"Извините, не удалось сгенерировать ответ на вопрос: {question}"
