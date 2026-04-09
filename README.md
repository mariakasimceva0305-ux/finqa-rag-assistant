# FinQA RAG Assistant

Finance-oriented **Retrieval-Augmented Generation (RAG)** pipeline for question answering over a domain corpus. The project combines corpus preparation, embedding-based retrieval, reranking, and answer generation grounded in retrieved context.

## Why This Project
Pure generation is brittle in domain-specific QA: it may sound plausible while being poorly grounded. This repository demonstrates a safer pattern:
- retrieve evidence first,
- improve context relevance with reranking,
- generate only from selected context.

## Problem Statement
Answer finance-related questions using a document collection rather than relying on a model's parametric memory alone.

## Core Idea
The system is organized as a compact RAG pipeline:
1. load and preprocess domain documents;
2. build retrieval-ready representations;
3. retrieve relevant candidates for a question;
4. rerank them for better context precision;
5. generate an answer grounded in the retrieved evidence.

## Repository Structure
```text
src/
  config.py             # pipeline configuration
  data_loader.py        # data ingestion and corpus preparation
  embedding_service.py  # embedding generation / vector representations
  generator.py          # answer generation layer
  pipeline.py           # orchestration of the full RAG flow
  retriever.py          # retrieval logic
baseline.py             # baseline run / comparison entrypoint
main.py                 # project entrypoint
questions.csv           # evaluation / input questions
train_data.csv          # project data
.env.example
requirements.txt
README.md
```

## Pipeline Design
### 1) Corpus Preparation
Documents are loaded and transformed into retrieval-ready chunks.

### 2) Retrieval
Relevant context is selected using embedding-based retrieval.

### 3) Reranking
A second scoring step improves the quality of the final context passed to generation.

### 4) Generation
The answer is generated from the retrieved context rather than from open-ended prompting alone.

## What Makes This Repository Useful
- modular RAG components instead of opaque notebook logic
- clear separation between retrieval and generation responsibilities
- baseline script for comparing a simpler setup
- finance-specific framing rather than generic QA

## Evaluation Design
The repository is most naturally evaluated along several dimensions:
- retrieval quality at top-K
- relevance of reranked context
- answer faithfulness to retrieved evidence
- end-to-end answer usefulness

For practical project reviews, I recommend tracking:
- `Recall@K` / hit rate for retrieval
- quality difference before vs after reranking
- qualitative groundedness checks
- latency per stage

## Running Locally
```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
pip install -r requirements.txt
python main.py
```

## Where This Design Is Strong
- domain QA where grounding matters
- explainable RAG demos for interviews / portfolio review
- experimentation with retrieval vs reranking contribution
- compact end-to-end architecture without unnecessary infrastructure

## Suggested Next Improvements
- add a formal evaluation table for baseline vs retrieval+rereanking pipeline
- document chunking strategy and top-K selection explicitly
- add citation-style answer formatting
- include failure cases and retrieval miss analysis
- separate offline evaluation script from demo entrypoints more clearly

## Limitations
- educational project scope rather than production retrieval scale
- answer quality depends strongly on corpus coverage
- no claim is made that generation should be trusted without context inspection
- gold QA annotations and benchmark-style evaluation can be extended further

## Takeaway
This repository shows a sound RAG design mindset: **retrieval first, reranking where helpful, and generation grounded in evidence**. That is the right default posture for domain QA systems.
