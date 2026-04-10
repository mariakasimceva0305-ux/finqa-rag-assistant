"""
Основной пайплайн RAG системы
"""

import pandas as pd
import time
from tqdm import tqdm
from typing import Optional
from .config import config
from .data_loader import DataLoader
from .embedding_service import EmbeddingService
from .retriever import Retriever
from .generator import AnswerGenerator


class FinancialQAPipeline:
    """Основной пайплайн для обработки финансовых вопросов"""
    
    def __init__(self, embedding_model: str = None, llm_model: str = None,
                 use_rerank: bool = True, chunk_size: int = None,
                 chunk_overlap: int = None, top_k: int = None):
        
        # Конфигурация
        if embedding_model:
            config.EMBEDDING_MODEL = embedding_model
        if llm_model:
            config.LLM_MODEL = llm_model
        
        self.use_rerank = use_rerank
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.top_k = top_k or config.TOP_K
        
        # Инициализация компонентов
        self.data_loader = DataLoader()
        self.embedding_service = EmbeddingService()
        self.retriever = Retriever(self.embedding_service)
        self.generator = AnswerGenerator()
        
        # Кэшированные данные
        self.embeddings = None
        self.chunks = None
        self.articles_data = None
        
        print(f"Инициализирован пайплайн с параметрами:")
        print(f"  Embedding модель: {config.EMBEDDING_MODEL}")
        print(f"  LLM модель: {config.LLM_MODEL}")
        print(f"  Реранкинг: {'включен' if use_rerank else 'выключен'}")
        print(f"  Размер чанка: {self.chunk_size}")
        print(f"  Перекрытие: {self.chunk_overlap}")
        print(f"  Top-K: {self.top_k}")
    
    def prepare_knowledge_base(self) -> None:
        """Подготовка базы знаний (эмбеддинги и чанки)"""
        print("Подготовка базы знаний...")
        
        # Загрузка статей
        articles_df = self.data_loader.load_articles()
        
        # Подготовка данных для эмбеддингов
        self.articles_data = self.data_loader.prepare_articles_for_embedding(articles_df)
        
        # Загрузка или создание эмбеддингов
        self.embeddings, self.chunks = self.embedding_service.load_or_create_embeddings(
            self.articles_data
        )
        
        print(f"База знаний готова: {len(self.chunks)} чанков")
    
    def answer_question(self, question: str) -> str:
        """Ответ на один вопрос"""
        # Проверяем, подготовлена ли база знаний
        if self.embeddings is None or self.chunks is None:
            self.prepare_knowledge_base()
        
        # Поиск релевантных чанков
        context_chunks = self.retriever.retrieve(
            question, 
            self.embeddings, 
            self.chunks,
            use_rerank=self.use_rerank
        )
        
        # Генерация ответа
        answer = self.generator.generate_answer(question, context_chunks)
        
        return answer
    
    def generate_submission(self, output_path: str = None) -> pd.DataFrame:
        """Генерация ответов для всех вопросов"""
        if output_path is None:
            output_path = config.OUTPUT_PATH
        
        # Загрузка вопросов
        questions_df = self.data_loader.load_questions()
        
        # Подготовка базы знаний
        if self.embeddings is None:
            self.prepare_knowledge_base()
        
        print(f"Генерация ответов для {len(questions_df)} вопросов...")
        
        answers = []
        start_time = time.time()
        
        for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df), 
                            desc="Генерация ответов"):
            question_id = row['id']
            question_text = str(row['question'])
            
            try:
                # Генерация ответа
                answer_text = self.answer_question(question_text)
                
                # Сохраняем результат
                answers.append({
                    'ID вопроса': question_id,
                    'Вопрос': question_text,
                    'Ответы на вопрос': answer_text
                })
                
                # Небольшая пауза для избежания rate limit
                time.sleep(0.05)
                
            except Exception as e:
                print(f"Ошибка при обработке вопроса {question_id}: {e}")
                answers.append({
                    'ID вопроса': question_id,
                    'Вопрос': question_text,
                    'Ответы на вопрос': f"Ошибка: {str(e)}"
                })
        
        # Создание DataFrame с результатами
        results_df = pd.DataFrame(answers)
        results_df = results_df.sort_values('ID вопроса')
        
        # Сохранение результатов
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        
        elapsed_time = time.time() - start_time
        print(f"Готово! Сохранено {len(results_df)} ответов в {output_path}")
        print(f"Общее время: {elapsed_time:.1f} секунд")
        print(f"Среднее время на вопрос: {elapsed_time/len(results_df):.1f} секунд")
        
        return results_df
