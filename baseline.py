#!/usr/bin/env python3
"""
Legacy-демо: прямой вызов LLM по строкам из questions.csv без RAG.

Офлайн-оценка и сравнение режимов — через ``python scripts/evaluate.py`` (см. README).
Этот скрипт оставлен для совместимости с простым CSV и отдельным submission.csv в корне.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# Тот же прокси-хост, что и в src/config (EMBEDDING_API_URL по умолчанию)
_DEFAULT_BASE_URL = "https://ai-for-finance-hack.up.railway.app/"
_DEFAULT_MODEL = "openrouter/mistralai/mistral-small-3.2-24b-instruct"


def _normalize_questions_df(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Приводит таблицу к колонкам id + question (как в ``DataLoader.load_questions``).

    Returns:
        work: колонки id, question
        answer_column: имя колонки для ответов в итоговом CSV
    """
    if "ID вопроса" in df.columns and "Вопрос" in df.columns:
        work = df.rename(columns={"ID вопроса": "id", "Вопрос": "question"})[["id", "question"]].copy()
        return work, "Ответы на вопрос"
    if "id" in df.columns and "question" in df.columns:
        work = df[["id", "question"]].copy()
        return work, "answer"
    raise ValueError(
        "Неверный формат questions.csv. Ожидаются колонки "
        "('id', 'question') либо legacy ('ID вопроса', 'Вопрос'). "
        f"Фактические колонки: {list(df.columns)}"
    )


def answer_generation(question: str, *, model: str | None = None) -> str:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Задайте LLM_API_KEY в окружении или в .env (как для основного пайплайна).")

    client = OpenAI(base_url=_DEFAULT_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=model or _DEFAULT_MODEL,
        messages=[{"role": "user", "content": [{"type": "text", "text": f"Ответь на вопрос: {question}"}]}],
    )
    return response.choices[0].message.content or ""


def main() -> None:
    root = Path(__file__).resolve().parent
    questions_path = root / "questions.csv"
    if not os.getenv("LLM_API_KEY"):
        print("LLM_API_KEY не задан. Укажите в .env или в окружении.", file=sys.stderr)
        sys.exit(1)
    if not questions_path.is_file():
        print(f"Файл не найден: {questions_path}", file=sys.stderr)
        sys.exit(1)

    original = pd.read_csv(questions_path)
    try:
        work, answer_col = _normalize_questions_df(original)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    answers: list[str] = []
    for q in tqdm(work["question"].astype(str).tolist(), desc="LLM"):
        answers.append(answer_generation(q))

    out = original.copy()
    out[answer_col] = answers

    out_path = root / "submission.csv"
    out.to_csv(out_path, index=False)
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
