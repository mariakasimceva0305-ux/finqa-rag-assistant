# FinQA-RAG-Assistant

Educational RAG pipeline for financial question answering.

## Repository Contents

- `main.py` - main run entrypoint.
- `baseline.py` - baseline implementation.
- `src/data_loader.py` - dataset loading and preprocessing.
- `src/embedding_service.py` - embedding generation and management.
- `src/retriever.py` - retrieval and reranking logic.
- `src/generator.py` - answer generation.
- `src/pipeline.py` - end-to-end pipeline orchestration.
- `questions.csv` and `train_data.csv` - input datasets.

## Implemented Functionality

The code implements a standard RAG flow:

- data chunk preparation,
- embedding-based retrieval,
- reranking stage,
- response generation over retrieved financial context.
