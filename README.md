# finqa-rag-assistant

Educational RAG pipeline for financial question answering.

## Project Scope

End-to-end retrieval-augmented generation flow over a financial document dataset.

## Repository Structure

- `main.py`
- `baseline.py`
- `src/data_loader.py`
- `src/embedding_service.py`
- `src/retriever.py`
- `src/generator.py`
- `src/pipeline.py`
- `questions.csv`
- `train_data.csv`

## Implemented Functionality

- document chunk preparation
- embedding-based retrieval
- reranking stage
- answer generation from retrieved context