#!/usr/bin/env python3
"""
Offline-style benchmark: latency, success rate, avg context size.
Без gold labels для retrieval не считает HitRate/MRR/Recall (см. README).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import config  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402
from src.pipeline import FinancialQAPipeline  # noqa: E402


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def run_mode(mode: str, output_dir: Path, max_questions: int | None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    use_rerank = mode == "full"
    metrics_name = "full_metrics.json" if use_rerank else "baseline_metrics.json"
    examples_name = "full_examples.csv" if use_rerank else "baseline_examples.csv"

    try:
        config.validate()
    except ValueError as e:
        payload = {
            "mode": mode,
            "error": str(e),
            "questions_total": 0,
            "success_rate": 0.0,
            "note": "Задайте LLM_API_KEY и EMBEDDER_API_KEY (например в .env) и повторите запуск.",
        }
        (output_dir / metrics_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / examples_name).write_text("id,question,ok,error\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    pipeline = FinancialQAPipeline(use_rerank=use_rerank)
    pipeline.prepare_knowledge_base()

    df = DataLoader.load_questions()
    if max_questions is not None:
        df = df.head(max_questions)

    totals: list[float] = []
    retr_ms: list[float] = []
    rerank_ms: list[float] = []
    gen_s: list[float] = []
    chunk_counts: list[int] = []
    successes = 0
    rows: list[dict] = []

    embeddings = pipeline.embeddings
    chunks = pipeline.chunks
    assert embeddings is not None and chunks is not None

    for _, row in df.iterrows():
        qid = row["id"]
        qtext = str(row["question"])
        t0 = time.perf_counter()

        try:
            t_r0 = time.perf_counter()
            scored = pipeline.retriever.search_relevant_chunks(
                qtext, embeddings, chunks, top_k=pipeline.top_k
            )
            t_r1 = time.perf_counter()
            retr_ms.append((t_r1 - t_r0) * 1000.0)

            t_rr0 = time.perf_counter()
            if use_rerank and scored:
                scored = pipeline.retriever.rerank_chunks(qtext, scored)
            t_rr1 = time.perf_counter()
            rerank_ms.append((t_rr1 - t_rr0) * 1000.0 if use_rerank else 0.0)

            context = [c for c, _ in scored]
            chunk_counts.append(len(context))

            t_g0 = time.perf_counter()
            answer = pipeline.generator.generate_answer(qtext, context)
            t_g1 = time.perf_counter()
            gen_s.append(t_g1 - t_g0)

            t1 = time.perf_counter()
            totals.append(t1 - t0)

            ok = bool(answer) and not str(answer).lower().startswith("ошибка")
            if ok:
                successes += 1
            rows.append(
                {
                    "id": qid,
                    "question": qtext[:500],
                    "ok": ok,
                    "latency_total_s": round(t1 - t0, 4),
                    "n_chunks": len(context),
                    "answer_preview": (answer[:300] + "…") if len(answer) > 300 else answer,
                }
            )
        except Exception as exc:  # noqa: BLE001
            t1 = time.perf_counter()
            totals.append(t1 - t0)
            retr_ms.append(0.0)
            rerank_ms.append(0.0)
            gen_s.append(0.0)
            chunk_counts.append(0)
            rows.append({"id": qid, "question": qtext[:500], "ok": False, "error": str(exc)})

    n = len(df)
    success_rate = successes / n if n else 0.0
    totals_sorted = sorted(totals)

    rerank_mean = None
    if use_rerank and rerank_ms:
        rerank_mean = round(statistics.mean(rerank_ms), 4)

    payload = {
        "mode": mode,
        "use_rerank": use_rerank,
        "questions_total": n,
        "success_rate": round(success_rate, 6),
        "latency_total_mean_s": round(statistics.mean(totals), 6) if totals else 0.0,
        "latency_total_p50_s": round(_percentile(totals_sorted, 50), 6),
        "latency_total_p95_s": round(_percentile(totals_sorted, 95), 6),
        "latency_retrieval_mean_ms": round(statistics.mean(retr_ms), 4) if retr_ms else 0.0,
        "latency_rerank_mean_ms": rerank_mean,
        "latency_generation_mean_s": round(statistics.mean(gen_s), 6) if gen_s else 0.0,
        "avg_context_chunks": round(statistics.mean(chunk_counts), 4) if chunk_counts else 0.0,
        "retrieval_quality_metrics": "not_computed_no_gold_labels",
    }

    (output_dir / metrics_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    one_md = [
        f"# Eval summary (`{mode}`)",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "```",
        "",
        "Retrieval quality (HitRate/MRR/Recall): not computed — no gold doc id in `questions.csv`.",
        "",
    ]
    (output_dir / "README_metrics.md").write_text("\n".join(one_md), encoding="utf-8")

    with (output_dir / examples_name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id", "question", "ok"])
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def _write_combined_readme(output_dir: Path, baseline: dict, full: dict) -> None:
    lines = [
        "# Eval summary (baseline vs full)",
        "",
        "## baseline (без rerank)",
        "",
        "```json",
        json.dumps(baseline, ensure_ascii=False, indent=2),
        "```",
        "",
        "## full (с rerank)",
        "",
        "```json",
        json.dumps(full, ensure_ascii=False, indent=2),
        "```",
        "",
        "Метрики качества retrieval (HitRate/MRR/Recall) не считаются: в `questions.csv` нет эталонного идентификатора документа.",
        "",
    ]
    (output_dir / "README_metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="FinQA latency / success benchmark")
    p.add_argument("--mode", choices=("baseline", "full", "both"), required=True)
    p.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    p.add_argument("--max-questions", type=int, default=None, help="Ограничить число вопросов для быстрого прогона")
    args = p.parse_args()
    if args.mode == "both":
        b = run_mode("baseline", args.output_dir, args.max_questions)
        f = run_mode("full", args.output_dir, args.max_questions)
        _write_combined_readme(args.output_dir, b, f)
        print("Written", args.output_dir / "README_metrics.md")
    else:
        run_mode(args.mode, args.output_dir, args.max_questions)


if __name__ == "__main__":
    main()
