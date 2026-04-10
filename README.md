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

Зависимости: `pip install -r requirements.txt` (для минимального запуска достаточно `openai`, `python-dotenv`, `pandas`, `numpy`, `requests`, `tqdm`).

Команды выполняются **из корня репозитория**. Код лежит в пакете `src/`; `main.py` добавляет корень в `sys.path` и импортирует `src.pipeline`, `src.config`.

### Без ключей API

`LLM_API_KEY` и `EMBEDDER_API_KEY` не заданы:

- `python main.py --test` или `--generate` — после `config.validate()` будет сообщение об ошибке и ненулевой код выхода (ожидаемо).
- `python scripts/evaluate.py --mode baseline|full|both` — не падает по импортам; в `reports/*_metrics.json` пишется ошибка валидации, `questions_total: 0`, без выдуманных latency.
- `python baseline.py` — сразу завершится с сообщением об отсутствии `LLM_API_KEY` (ожидаемо).

### С ключами API

В `.env` или в окружении заданы оба ключа:

- `python main.py --test` — три тестовых вопроса без `questions.csv`.
- `python main.py --generate` — все строки из `questions.csv`.
- `python scripts/evaluate.py --mode baseline` — замеры без rerank.
- `python scripts/evaluate.py --mode full` — с rerank.
- `python scripts/evaluate.py --mode both` — оба режима и общий `reports/README_metrics.md`.

Флаг `--max-questions N` ограничивает число вопросов в evaluator.

`python baseline.py` — legacy: прямой вызов LLM по `questions.csv` в корне (те же схемы колонок, что и у `DataLoader`: `ID вопроса`/`Вопрос` или `id`/`question`); результат — `submission.csv` в корне. Нужен `LLM_API_KEY`; для офлайн-оценки RAG используйте `scripts/evaluate.py`.

## Результаты

Артефакты пишет `scripts/evaluate.py` в каталог `--output-dir` (по умолчанию `reports/`):

| Файл | Содержание |
|------|------------|
| `reports/baseline_metrics.json` | режим без rerank |
| `reports/full_metrics.json` | режим с rerank |
| `reports/baseline_examples.csv` | примеры по вопросам |
| `reports/full_examples.csv` | примеры по вопросам |
| `reports/README_metrics.md` | сводка (`--mode both` перезаписывает одним файлом оба режима) |

Без ключей в JSON честно указывается причина и `questions_total: 0`. С ключами там же появляются `latency_*`, `success_rate`, `avg_context_chunks`.

Поле `retrieval_quality_metrics` остаётся `not_computed_no_gold_labels`: в `questions.csv` нет эталонного идентификатора документа для retrieval-метрик.

## Ограничения

- без ключей API пайплайн и evaluator не выполняются;
- качество генерации зависит от полноты найденных фрагментов и доступности внешних API;
- `baseline.py` и основной контур читают один и тот же файл `questions.csv` в корне (путь по умолчанию — в `src/config.py`); для baseline нужен только `LLM_API_KEY` в окружении / `.env`, иначе скрипт завершится с ошибкой.
