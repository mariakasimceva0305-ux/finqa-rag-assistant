#!/usr/bin/env python3
"""
FinAnswer: RAG-система для финансового консультирования
Точка входа для генерации ответов на финансовые вопросы
"""

import argparse
import os
import sys
from pathlib import Path

# Корень репозитория в PYTHONPATH — пакет `src`
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import config
from src.pipeline import FinancialQAPipeline


def main():
    parser = argparse.ArgumentParser(
        description="FinAnswer - RAG система для финансового консультирования",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --generate                     # Генерация ответов для всех вопросов
  python main.py --test                         # Тест на 3 вопросах
  python main.py --generate --no_rerank         # Без реранкинга
  python main.py --model "llama-3-70b"          # С указанием модели
        """
    )
    
    parser.add_argument("--generate", action="store_true",
                       help="Сгенерировать ответы для всех вопросов")
    parser.add_argument("--test", action="store_true",
                       help="Запустить тест на 3 вопросах")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small",
                       help="Модель для эмбеддингов")
    parser.add_argument("--llm_model", type=str, default="openrouter/meta-llama/llama-3-70b-instruct",
                       help="LLM модель для генерации")
    parser.add_argument("--no_rerank", action="store_true",
                       help="Отключить реранкинг")
    parser.add_argument("--chunk_size", type=int, default=800,
                       help="Размер чанков для разбиения текста")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                       help="Перекрытие между чанками")
    parser.add_argument("--top_k", type=int, default=3,
                       help="Количество релевантных чанков для поиска")
    parser.add_argument("--output", type=str, default="./data/processed/submission.csv",
                       help="Путь для сохранения результатов")
    
    args = parser.parse_args()
    
    if not (args.generate or args.test):
        parser.print_help()
        return

    config.validate()

    print("=" * 60)
    print("FinAnswer: RAG-система для финансового консультирования")
    print("=" * 60)
    
    try:
        pipeline = FinancialQAPipeline(
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            use_rerank=not args.no_rerank,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k
        )
        
        if args.test:
            print("Запуск тестового режима...")
            test_questions = [
                "Что такое облигации и как они работают?",
                "Как открыть депозит в банке?",
                "Какие налоги платят при продаже недвижимости?"
            ]
            for i, question in enumerate(test_questions, 1):
                print(f"\nТестовый вопрос {i}: {question}")
                answer = pipeline.answer_question(question)
                print(f"Ответ: {answer[:200]}..." if len(answer) > 200 else f"Ответ: {answer}")
        
        if args.generate:
            print("Запуск генерации ответов для всех вопросов...")
            pipeline.generate_submission(output_path=args.output)
    
    except Exception as e:
        print(f"Ошибка при выполнении: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
