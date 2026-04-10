"""
Загрузка и предобработка данных
"""

import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from .config import config


class DataLoader:
    """Загрузчик и предобработчик данных"""
    
    @staticmethod
    def load_articles(file_path: str = None) -> pd.DataFrame:
        """Загрузка статей из CSV файла"""
        if file_path is None:
            file_path = config.TRAIN_DATA_PATH
        
        print(f"Загрузка статей из {file_path}...")
        df = pd.read_csv(file_path)
        
        # Проверка обязательных колонок
        required_columns = ['id', 'text']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует обязательная колонка: {col}")
        
        print(f"Загружено {len(df)} статей")
        return df
    
    @staticmethod
    def load_questions(file_path: str = None) -> pd.DataFrame:
        """Загрузка вопросов из CSV файла"""
        if file_path is None:
            file_path = config.QUESTIONS_PATH
        
        print(f"Загрузка вопросов из {file_path}...")
        df = pd.read_csv(file_path)
        
        # Проверка и переименование колонок
        if 'ID вопроса' in df.columns and 'Вопрос' in df.columns:
            df = df.rename(columns={'ID вопроса': 'id', 'Вопрос': 'question'})
        
        if 'id' not in df.columns or 'question' not in df.columns:
            raise ValueError("CSV должен содержать колонки 'id' и 'question'")
        
        print(f"Загружено {len(df)} вопросов")
        return df
    
    @staticmethod
    def split_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Разбиение текста на чанки с перекрытием"""
        if chunk_size is None:
            chunk_size = config.CHUNK_SIZE
        if overlap is None:
            overlap = config.CHUNK_OVERLAP
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end].strip()
            
            if chunk:  # Пропускаем пустые чанки
                chunks.append(chunk)
            
            start = end - overlap
            if start >= text_length:
                break
        
        return chunks
    
    @staticmethod
    def prepare_articles_for_embedding(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
        """Подготовка статей для создания эмбеддингов"""
        articles_data = []
        
        print("Подготовка статей для эмбеддингов...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Обработка статей"):
            article_id = str(row['id'])
            text = str(row['text'])
            annotation = str(row.get('annotation', ''))
            
            # Объединяем аннотацию и текст
            full_text = f"{annotation}\n\n{text}" if annotation else text
            
            # Разбиваем на чанки
            chunks = DataLoader.split_text(full_text)
            
            # Сохраняем каждый чанк с идентификатором
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{article_id}_chunk_{chunk_idx}"
                articles_data.append((chunk_id, chunk, article_id))
        
        print(f"Создано {len(articles_data)} чанков из {len(df)} статей")
        return articles_data
