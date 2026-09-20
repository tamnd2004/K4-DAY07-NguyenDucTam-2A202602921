from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    HeadingChunker,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    RecursiveChunker,
    SentenceChunker,
    _mock_embed,
)


STRATEGY_NAMES = ("fixed_size", "by_sentences", "recursive", "heading")


@dataclass(frozen=True)
class CorpusDocument:
    metadata: dict[str, str]
    content: str


@dataclass(frozen=True)
class BenchmarkCase:
    question: str
    gold_answer: str
    expected_doc_id: str
    evidence: str
    metadata_filter: dict[str, Any] | None


@dataclass(frozen=True)
class RetrievalEvaluation:
    document_rank: int | None
    evidence_rank: int | None
    score: int


class TeeOutput:
    """Write benchmark output to both the terminal and a report file."""

    def __init__(self, terminal: Any, report_file: Any) -> None:
        self.terminal = terminal
        self.report_file = report_file

    def write(self, data: str) -> int:
        self.terminal.write(data)
        self.report_file.write(data)
        self.report_file.flush()
        return len(data)

    def flush(self) -> None:
        self.terminal.flush()
        self.report_file.flush()

    def isatty(self) -> bool:
        return False


class BatchEmbedder:
    """Batch and memoize embeddings while preserving the scalar callable API."""

    def __init__(self, backend: Any, provider: str, batch_size: int) -> None:
        self.backend = backend
        self.provider = provider
        self.batch_size = batch_size
        self.cache: dict[str, list[float]] = {}

    def __call__(self, text: str) -> list[float]:
        if text not in self.cache:
            self.preload([text])
        return self.cache[text]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "local":
            vectors = self.backend.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return [
                [float(value) for value in vector]
                for vector in vectors
            ]

        if self.provider == "openai":
            response = self.backend.client.embeddings.create(
                model=self.backend.model_name,
                input=texts,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            return [
                [float(value) for value in item.embedding]
                for item in ordered
            ]

        if self.provider == "gemini":
            response = self.backend.client.models.embed_content(
                model=self.backend.model_name,
                contents=texts,
            )
            return [
                [float(value) for value in embedding.values]
                for embedding in response.embeddings
            ]

        return [self.backend(text) for text in texts]

    def preload(self, texts: list[str]) -> None:
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if not missing:
            print("  All embeddings served from the in-run cache", flush=True)
            return

        total_batches = (len(missing) + self.batch_size - 1) // self.batch_size
        for batch_number, start in enumerate(
            range(0, len(missing), self.batch_size), start=1
        ):
            batch = missing[start : start + self.batch_size]
            vectors = self._embed_batch(batch)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding backend returned {len(vectors)} vectors for "
                    f"{len(batch)} inputs"
                )
            self.cache.update(zip(batch, vectors))
            print(
                f"  Embedded batch {batch_number}/{total_batches} "
                f"({min(start + len(batch), len(missing))}/{len(missing)} new texts)",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    load_dotenv(override=False)
    default_provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()

    parser = argparse.ArgumentParser(
        description="Compare retrieval quality across the supported chunking strategies."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/ecommerce"),
        help="Corpus directory containing sources.csv and Markdown files",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="Benchmark JSON path (default: <data-dir>/benchmark.json)",
    )
    parser.add_argument(
        "--strategy",
        choices=("all", *STRATEGY_NAMES),
        default="all",
        help="Run all strategies sequentially or only one strategy",
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "local", "openai", "gemini"),
        default=(
            default_provider
            if default_provider in {"mock", "local", "openai", "gemini"}
            else "mock"
        ),
        help=f"Embedding backend (default: ${EMBEDDING_PROVIDER_ENV} or mock)",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of results per query"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Character limit for fixed/recursive/heading chunks",
    )
    parser.add_argument(
        "--overlap", type=int, default=50, help="Fixed-size chunk overlap"
    )
    parser.add_argument(
        "--sentences-per-chunk",
        type=int,
        default=3,
        help="Maximum sentence count for sentence chunking",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size for local/API providers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ket_qua_benchmark.txt"),
        help="Text report path (default: ket_qua_benchmark.txt)",
    )
    return parser.parse_args()


def clean_frontmatter_value(value: str) -> str:
    value = re.sub(r"\s+#.*$", "", value)
    return value.strip().strip("\"'")


def read_markdown_document(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) < 3 or parts[0].strip():
        raise ValueError(f"Missing YAML frontmatter in {path}")

    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key:
            metadata[key] = clean_frontmatter_value(value)

    return metadata, parts[2].strip()


def load_corpus(data_dir: Path) -> list[CorpusDocument]:
    manifest_path = data_dir / "sources.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        rows = list(csv.DictReader(manifest_file))

    if not rows:
        raise ValueError(f"Manifest has no document rows: {manifest_path}")

    corpus: list[CorpusDocument] = []
    seen_doc_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        doc_id = (row.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError(f"Missing doc_id in {manifest_path}, row {row_number}")
        if doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate doc_id in manifest: {doc_id}")
        seen_doc_ids.add(doc_id)

        # Resolve by doc_id instead of file_path so Windows-style separators in
        # an existing manifest do not break a run on Linux.
        path = data_dir / f"{doc_id}.md"
        if not path.is_file():
            raise FileNotFoundError(f"Document listed in manifest was not found: {path}")

        metadata, content = read_markdown_document(path)
        if not content:
            raise ValueError(f"Document content is empty: {path}")
        if metadata.get("doc_id") != doc_id:
            raise ValueError(
                f"doc_id mismatch for {path}: frontmatter={metadata.get('doc_id')!r}, "
                f"manifest={doc_id!r}"
            )

        metadata["doc_id"] = doc_id
        metadata["source"] = str(path)
        corpus.append(CorpusDocument(metadata=metadata, content=content))

    return corpus


def require_text(value: Any, field: str, case_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Benchmark case {case_number}: {field} must be a non-empty string"
        )
    return value.strip()


def load_benchmarks(path: Path) -> list[BenchmarkCase]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}\n"
            "Create a JSON array with exactly five cases containing question, gold_answer, "
            "expected_doc_id, evidence, and optional metadata_filter."
        )

    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list) or len(raw_cases) != 5:
        raise ValueError("Benchmark JSON must be an array containing exactly five cases")

    cases: list[BenchmarkCase] = []
    for case_number, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Benchmark case {case_number} must be a JSON object")

        metadata_filter = raw_case.get("metadata_filter")
        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            raise ValueError(
                f"Benchmark case {case_number}: metadata_filter must be an object or null"
            )

        cases.append(
            BenchmarkCase(
                question=require_text(raw_case.get("question"), "question", case_number),
                gold_answer=require_text(
                    raw_case.get("gold_answer"), "gold_answer", case_number
                ),
                expected_doc_id=require_text(
                    raw_case.get("expected_doc_id"), "expected_doc_id", case_number
                ),
                evidence=require_text(raw_case.get("evidence"), "evidence", case_number),
                metadata_filter=metadata_filter,
            )
        )
    return cases


def build_embedder(provider: str, batch_size: int) -> BatchEmbedder:
    if provider == "local":
        backend = LocalEmbedder(
            model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL)
        )
    elif provider == "openai":
        backend = OpenAIEmbedder(
            model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL)
        )
    elif provider == "gemini":
        backend = GeminiEmbedder(
            model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL)
        )
    else:
        print(
            "WARNING: MockEmbedder does not encode semantic meaning. "
            "Use it only to verify the benchmark pipeline."
        )
        backend = _mock_embed
    return BatchEmbedder(backend, provider, batch_size)


def make_strategies(args: argparse.Namespace) -> dict[str, Any]:
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than zero")
    if args.overlap < 0 or args.overlap >= args.chunk_size:
        raise ValueError("--overlap must satisfy 0 <= overlap < chunk-size")
    if args.sentences_per_chunk <= 0:
        raise ValueError("--sentences-per-chunk must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    strategies = {
        "fixed_size": FixedSizeChunker(
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        ),
        "by_sentences": SentenceChunker(
            max_sentences_per_chunk=args.sentences_per_chunk,
        ),
        "recursive": RecursiveChunker(chunk_size=args.chunk_size),
        "heading": HeadingChunker(chunk_size=args.chunk_size),
    }
    if args.strategy == "all":
        return strategies
    return {args.strategy: strategies[args.strategy]}


def build_store(
    strategy_name: str,
    chunker: Any,
    corpus: list[CorpusDocument],
    embedder: Any,
) -> tuple[EmbeddingStore, int]:
    documents: list[Document] = []
    for source in corpus:
        chunks = chunker.chunk(source.content)
        for chunk_index, chunk in enumerate(chunks):
            documents.append(
                Document(
                    id=f"{source.metadata['doc_id']}#{chunk_index}",
                    content=chunk,
                    metadata=dict(source.metadata),
                )
            )

    store = EmbeddingStore(
        collection_name=f"benchmark-{strategy_name}",
        embedding_fn=embedder,
    )
    total = len(documents)
    print(f"Prepared {total} chunks. Embedding them now...", flush=True)
    embedder.preload([document.content for document in documents])
    print("Storing embedded chunks...", flush=True)
    for index, document in enumerate(documents, start=1):
        store.add_documents([document])
        if index == total:
            print(f"  Stored {index}/{total} chunks", flush=True)
    return store, len(documents)


def normalize(text: str) -> str:
    # Parenthetical number spellings such as ``15 (mười lăm) ngày`` should not
    # make an otherwise exact answer marker fail to match ``15 ngày``.
    without_parentheticals = re.sub(r"\([^)]*\)", " ", text.casefold())
    return " ".join(without_parentheticals.split())


def is_relevant(result: dict[str, Any], case: BenchmarkCase) -> bool:
    # Content-level relevance is deliberately independent of doc_id.  The same
    # answer can be repeated in another authoritative document; requiring both
    # the gold doc and the evidence would merely disguise document-level
    # scoring as chunk-level scoring.
    return normalize(case.evidence) in normalize(result["content"])


def evaluate_results(
    results: list[dict[str, Any]], case: BenchmarkCase
) -> RetrievalEvaluation:
    document_rank = next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if result["metadata"].get("doc_id") == case.expected_doc_id
        ),
        None,
    )
    evidence_rank = next(
        (
            rank
            for rank, result in enumerate(results, start=1)
            if is_relevant(result, case)
        ),
        None,
    )
    score = 2 if evidence_rank == 1 else 1 if evidence_rank is not None else 0
    return RetrievalEvaluation(
        document_rank=document_rank,
        evidence_rank=evidence_rank,
        score=score,
    )


def print_results(
    label: str,
    results: list[dict[str, Any]],
    case: BenchmarkCase,
) -> RetrievalEvaluation:
    evaluation = evaluate_results(results, case)
    print(f"{label}:")
    print(
        "  Gold document rank: "
        f"{evaluation.document_rank if evaluation.document_rank is not None else 'NOT FOUND'}"
    )
    print(
        "  Evidence chunk rank: "
        f"{evaluation.evidence_rank if evaluation.evidence_rank is not None else 'NOT FOUND'}"
    )
    print(f"  Rubric score: {evaluation.score}/2")

    if not results:
        print("  No retrieval results.")
        return evaluation

    for rank, result in enumerate(results, start=1):
        marker = "MATCH" if is_relevant(result, case) else "----"
        doc_id = result["metadata"].get("doc_id", "unknown")
        chunk_id = result.get("id", "unknown")
        preview = " ".join(result["content"].split())[:180].rstrip()
        print(
            f"  {rank}. [{marker}] score={result['score']:.4f} "
            f"doc={doc_id} chunk={chunk_id}\n"
            f"     {preview}"
        )
    return evaluation


def compare_ab(
    filtered_results: list[dict[str, Any]],
    filtered_evaluation: RetrievalEvaluation,
    unfiltered_results: list[dict[str, Any]],
    unfiltered_evaluation: RetrievalEvaluation,
) -> str:
    filtered_ids = [result.get("id") for result in filtered_results]
    unfiltered_ids = [result.get("id") for result in unfiltered_results]
    if filtered_ids == unfiltered_ids:
        return "NO EFFECT: filtered and unfiltered top-k are identical"

    filtered_rank = filtered_evaluation.evidence_rank or float("inf")
    unfiltered_rank = unfiltered_evaluation.evidence_rank or float("inf")
    if filtered_rank < unfiltered_rank:
        return "IMPROVED: filter moved answer evidence higher or into top-k"
    if filtered_rank > unfiltered_rank:
        return "WORSE: filter reduced recall for the answer evidence"
    return "CHANGED: top-k changed, but the evidence rank did not improve"


def explain_failure(
    evaluation: RetrievalEvaluation,
    case: BenchmarkCase,
    ab_result: str | None,
) -> tuple[str, str]:
    if ab_result and ab_result.startswith("WORSE"):
        return (
            "Metadata filter excluded or demoted the chunk containing the answer.",
            "Revisit the audience labels or relax the hard filter to protect recall.",
        )
    if evaluation.document_rank is not None and evaluation.evidence_rank is None:
        return (
            "The gold document reached top-k, but the returned section did not contain the answer evidence.",
            "Improve chunk boundaries/overlap or use heading-aware chunks so the answer section has more chances to rank.",
        )
    if evaluation.evidence_rank is None:
        return (
            "Neither an answer-bearing chunk nor sufficient evidence reached top-k.",
            "Use a semantic embedding model, inspect query wording, and verify that chunking did not split the evidence.",
        )
    return (
        f"The answer-bearing chunk ranked at position {evaluation.evidence_rank}, not top-1.",
        "Inspect the higher-ranked chunks and tune chunk size, overlap, or query wording to reduce topical noise.",
    )


def run_strategy(
    strategy_name: str,
    chunker: Any,
    corpus: list[CorpusDocument],
    cases: list[BenchmarkCase],
    embedder: Any,
    top_k: int,
) -> dict[str, Any]:
    print("\n" + "=" * 88)
    print(f"STRATEGY: {strategy_name}")
    print("=" * 88)

    store, chunk_count = build_store(strategy_name, chunker, corpus, embedder)
    print(f"Documents: {len(corpus)} | Chunks: {chunk_count} | Top-k: {top_k}")

    document_hit_count = 0
    evidence_hit_count = 0
    total_score = 0
    reciprocal_rank_sum = 0.0
    ab_checks: list[str] = []
    failures: list[dict[str, str | int]] = []
    for case_number, case in enumerate(cases, start=1):
        filtered_results = store.search_with_filter(
            case.question,
            top_k=top_k,
            metadata_filter=case.metadata_filter,
        )

        print(f"\nQ{case_number}: {case.question}")
        print(f"Gold: {case.gold_answer}")
        print(f"Expected doc: {case.expected_doc_id}")
        print(f"Filter: {case.metadata_filter}")
        label = "FILTERED TOP-K" if case.metadata_filter else "TOP-K"
        evaluation = print_results(label, filtered_results, case)

        ab_result: str | None = None
        if case.metadata_filter:
            unfiltered_results = store.search_with_filter(
                case.question,
                top_k=top_k,
                metadata_filter=None,
            )
            unfiltered_evaluation = print_results(
                "UNFILTERED TOP-K", unfiltered_results, case
            )
            ab_result = compare_ab(
                filtered_results,
                evaluation,
                unfiltered_results,
                unfiltered_evaluation,
            )
            ab_checks.append(f"Q{case_number}: {ab_result}")
            print(f"A/B verdict: {ab_result}")

        if evaluation.document_rank is not None:
            document_hit_count += 1
        if evaluation.evidence_rank is not None:
            evidence_hit_count += 1
            reciprocal_rank_sum += 1 / evaluation.evidence_rank
        total_score += evaluation.score

        print(
            "Agent answer correctness: MANUAL CHECK REQUIRED against the gold answer "
            "(this runner evaluates retrieval evidence, not text generation)."
        )
        if evaluation.score < 2:
            reason, suggestion = explain_failure(evaluation, case, ab_result)
            failures.append(
                {
                    "question_number": case_number,
                    "question": case.question,
                    "score": evaluation.score,
                    "reason": reason,
                    "suggestion": suggestion,
                }
            )

    mean_reciprocal_rank = reciprocal_rank_sum / len(cases)
    print(
        f"\nSUMMARY {strategy_name}: chunks={chunk_count}, "
        f"DocHit@{top_k}={document_hit_count}/{len(cases)}, "
        f"EvidenceHit@{top_k}={evidence_hit_count}/{len(cases)}, "
        f"Score={total_score}/10, MRR={mean_reciprocal_rank:.3f}"
    )
    if ab_checks:
        print("A/B FILTER SUMMARY:")
        for check in ab_checks:
            print(f"  - {check}")

    if failures:
        failure = failures[0]
        print("FAILURE CASE TO REPORT:")
        print(f"  Question: Q{failure['question_number']} - {failure['question']}")
        print(f"  Score: {failure['score']}/2")
        print(f"  Why: {failure['reason']}")
        print(f"  Proposed fix: {failure['suggestion']}")
    else:
        print("FAILURE CASE TO REPORT: none with score below 2 in this run.")

    return {
        "chunks": chunk_count,
        "document_hits": document_hit_count,
        "evidence_hits": evidence_hit_count,
        "score": total_score,
        "mrr": mean_reciprocal_rank,
        "failures": failures,
        "ab_checks": ab_checks,
    }


def run_benchmark(args: argparse.Namespace) -> int:
    benchmark_path = args.benchmark or args.data_dir / "benchmark.json"

    try:
        cases = load_benchmarks(benchmark_path)
        corpus = load_corpus(args.data_dir)
        strategies = make_strategies(args)
        print(f"Initializing embedding provider: {args.provider}...", flush=True)
        embedder = build_embedder(args.provider, args.batch_size)
    except (
        FileNotFoundError,
        ImportError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"ERROR: {error}")
        return 2

    known_doc_ids = {document.metadata["doc_id"] for document in corpus}
    unknown_doc_ids = sorted(
        {case.expected_doc_id for case in cases} - known_doc_ids
    )
    if unknown_doc_ids:
        print(
            "ERROR: Benchmark references doc_id values not present in sources.csv: "
            + ", ".join(unknown_doc_ids)
        )
        return 2

    print(
        f"Corpus: {args.data_dir} | Benchmark: {benchmark_path} | "
        f"Embedding provider: {args.provider}"
    )
    print(
        "Scoring has two retrieval levels: gold document rank and answer-evidence "
        "chunk rank. The 2/1/0 score uses the evidence rank."
    )

    summaries: dict[str, dict[str, Any]] = {}
    for strategy_name, chunker in strategies.items():
        summaries[strategy_name] = run_strategy(
            strategy_name,
            chunker,
            corpus,
            cases,
            embedder,
            args.top_k,
        )

    if len(summaries) > 1:
        print("\n" + "=" * 88)
        print("FINAL COMPARISON")
        print("=" * 88)
        for strategy_name, summary in summaries.items():
            print(
                f"{strategy_name:16} chunks={summary['chunks']:4} "
                f"DocHit@{args.top_k}={summary['document_hits']}/5 "
                f"EvidenceHit@{args.top_k}={summary['evidence_hits']}/5 "
                f"Score={summary['score']}/10 MRR={summary['mrr']:.3f}"
            )

    print(
        "\nNOTE: A complete rubric judgment still requires comparing the generated "
        "agent answer with Gold. This runner intentionally does not claim answer "
        "correctness when no generation LLM is configured."
    )

    return 0


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as report_file:
        tee = TeeOutput(sys.stdout, report_file)
        with redirect_stdout(tee):
            exit_code = run_benchmark(args)
            print(f"\nBenchmark report: {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
