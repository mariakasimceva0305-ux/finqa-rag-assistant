# FinQA RAG Assistant

Контур ответов на финансовые вопросы по статьям из `train_data.csv`: эмбеддинги, поиск чанков, опциональный rerank через API и генерация ответа LLM. Вопросы и статьи — файлы `questions.csv` и `train_data.csv` в корне (пути задаёт `src/config.py`).

## Назначение

По коллекции статей находить фрагменты, релевантные вопросу, и формировать ответ с опорой на найденный контекст.

## Архитектура

```text
Финансовые документы
        │
        ▼
Разбиение на фрагменты
        │
        ▼
Векторное представление
        │
        ▼
Поиск top-K фрагментов
        │
        ▼
Переранжирование (опционально)
        │
        ▼
Генерация ответа по найденным фрагментам
```

## Данные

- `train_data.csv` — статьи (`id`, `text`, …).
- `questions.csv` — вопросы (`id`, `question`); **нет** колонки с эталонным `doc_id`, поэтому **offline HitRate@K / MRR / Recall по retrieval в отчёте не считаются**.

## Запуск

Требуются `LLM_API_KEY` и `EMBEDDER_API_KEY` (переменные окружения или `.env` в корне).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt   # или минимально: openai python-dotenv pandas numpy requests tqdm

python main.py --test              # 3 встроенных вопроса
python main.py --generate          # все вопросы из questions.csv

python baseline.py                 # упрощённый контур только LLM, читает ./questions.csv (корень)
```

Оценка задержек и успешности (без метрик качества поиска по gold):

```bash
python scripts/evaluate.py --mode baseline --output-dir reports   # без rerank
python scripts/evaluate.py --mode full --output-dir reports         # с rerank
# или оба режима и общий README_metrics.md:
python scripts/evaluate.py --mode both --output-dir reports
```

Опция `--max-questions N` ограничивает число строк для быстрого прогона.

## Результаты

Артефакты создаёт только `scripts/evaluate.py`:

| Файл | Содержание |
|------|------------|
| `reports/baseline_metrics.json` | режим без rerank |
| `reports/full_metrics.json` | режим с rerank |
| `reports/baseline_examples.csv` | примеры по вопросам |
| `reports/full_examples.csv` | примеры по вопросам |
| `reports/README_metrics.md` | краткая сводка |

**Текущее состояние репозитория после последнего прогона в среде без ключей:** в JSON зафиксирована ошибка конфигурации (`questions_total: 0`). Чтобы получить численные метрики задержек и `success_rate`, задайте ключи и выполните `python scripts/evaluate.py --mode both` — значения появятся в тех же файлах.

Поле `retrieval_quality_metrics` в JSON всегда `not_computed_no_gold_labels`, пока в данных нет эталонных соответствий вопрос → документ.

## Ограничения

- без ключей API пайплайн и evaluator не выполняются;
- качество генерации зависит от полноты найденных фрагментов и доступности внешних API;
- `baseline.py` и основной контур используют `questions.csv` в корне (путь настраивается в `src/config.py`).
