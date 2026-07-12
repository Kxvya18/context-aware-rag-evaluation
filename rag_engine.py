

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)?")


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str
    tokens: list[str]


@dataclass
class SearchResult:
    chunk_id: str
    source: str
    score: float
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[SearchResult]
    groundedness_score: float


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def read_documents(doc_dir: Path) -> dict[str, str]:
    docs: dict[str, str] = {}
    for path in sorted(doc_dir.glob("*.txt")):
        docs[path.name] = path.read_text(encoding="utf-8")
    if not docs:
        raise FileNotFoundError(f"No .txt documents found in {doc_dir}")
    return docs


def chunk_text(source: str, text: str, max_tokens: int = 120, overlap: int = 24) -> list[Chunk]:
    words = tokenize(text)
    if not words:
        return []

    raw_tokens = text.split()
    chunks: list[Chunk] = []
    start = 0
    chunk_no = 1
    while start < len(raw_tokens):
        window = raw_tokens[start : start + max_tokens]
        chunk_text_value = " ".join(window)
        chunk_tokens = tokenize(chunk_text_value)
        chunks.append(
            Chunk(
                chunk_id=f"{Path(source).stem}-{chunk_no:03d}",
                source=source,
                text=chunk_text_value,
                tokens=chunk_tokens,
            )
        )
        if start + max_tokens >= len(raw_tokens):
            break
        start += max_tokens - overlap
        chunk_no += 1
    return chunks


class RagIndex:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.doc_freq = self._document_frequencies(chunks)
        self.idf = self._inverse_document_frequencies(self.doc_freq, len(chunks))
        self.chunk_vectors = [self._tfidf_vector(chunk.tokens) for chunk in chunks]

    @classmethod
    def from_directory(cls, doc_dir: str | Path, max_tokens: int = 120, overlap: int = 24) -> "RagIndex":
        documents = read_documents(Path(doc_dir))
        chunks: list[Chunk] = []
        for source, text in documents.items():
            chunks.extend(chunk_text(source, text, max_tokens=max_tokens, overlap=overlap))
        return cls(chunks)

    @staticmethod
    def _document_frequencies(chunks: Iterable[Chunk]) -> Counter:
        df: Counter = Counter()
        for chunk in chunks:
            df.update(set(chunk.tokens))
        return df

    @staticmethod
    def _inverse_document_frequencies(df: Counter, n_docs: int) -> dict[str, float]:
        return {term: math.log((1 + n_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}

    def _tfidf_vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = max(1, len(tokens))
        vector = {term: (count / total) * self.idf.get(term, 0.0) for term, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {term: value / norm for term, value in vector.items()}

    @staticmethod
    def _cosine(query_vec: dict[str, float], chunk_vec: dict[str, float]) -> float:
        if len(query_vec) > len(chunk_vec):
            query_vec, chunk_vec = chunk_vec, query_vec
        return sum(value * chunk_vec.get(term, 0.0) for term, value in query_vec.items())

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        query_vec = self._tfidf_vector(tokenize(query))
        scored = []
        for chunk, vector in zip(self.chunks, self.chunk_vectors):
            score = self._cosine(query_vec, vector)
            if score > 0:
                scored.append(SearchResult(chunk.chunk_id, chunk.source, score, chunk.text))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def answer(self, question: str, top_k: int = 4, max_sentences: int = 4) -> Answer:
        citations = self.search(question, top_k=top_k)
        if not citations:
            return Answer(
                question=question,
                answer="I could not find enough grounded information in the indexed documents to answer this question.",
                citations=[],
                groundedness_score=0.0,
            )

        query_terms = set(tokenize(question))
        candidate_sentences: list[tuple[float, str, SearchResult]] = []
        for result in citations:
            for sentence in sentence_split(result.text):
                sentence_terms = set(tokenize(sentence))
                overlap = len(query_terms & sentence_terms)
                if overlap:
                    score = result.score + overlap / max(1, len(query_terms))
                    candidate_sentences.append((score, sentence, result))

        candidate_sentences.sort(key=lambda item: item[0], reverse=True)
        selected: list[str] = []
        seen = set()
        for _, sentence, result in candidate_sentences:
            normalized = sentence.lower()
            if normalized in seen:
                continue
            selected.append(f"{sentence} [{result.source}]")
            seen.add(normalized)
            if len(selected) >= max_sentences:
                break

        if not selected:
            selected = [f"{citations[0].text.strip()} [{citations[0].source}]"]

        answer_text = " ".join(selected)
        groundedness = groundedness_score(answer_text, citations)
        return Answer(question=question, answer=answer_text, citations=citations, groundedness_score=groundedness)

    def save(self, path: str | Path) -> None:
        payload = {
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "doc_freq": dict(self.doc_freq),
            "idf": self.idf,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_documents(doc_dir: str | Path) -> dict[str, str]:
    return read_documents(Path(doc_dir))


def build_chunks(documents: dict[str, str], max_tokens: int = 120, overlap: int = 24) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in documents.items():
        chunks.extend(chunk_text(source, text, max_tokens=max_tokens, overlap=overlap))
    return chunks


def retrieve(question: str, chunks: list[Chunk], top_k: int = 3) -> list[tuple[float, Chunk]]:
    index = RagIndex(chunks)
    results = index.search(question, top_k=top_k)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    return [(result.score, chunk_by_id[result.chunk_id]) for result in results if result.chunk_id in chunk_by_id]


def answer_question(question: str, doc_dir: str | Path = "sample_docs", top_k: int = 3) -> dict:
    index = RagIndex.from_directory(doc_dir)
    answer = index.answer(question, top_k=top_k)
    return {
        "question": answer.question,
        "answer": answer.answer,
        "groundedness_score": answer.groundedness_score,
        "sources": [
            {
                "source": citation.source,
                "chunk_id": citation.chunk_id,
                "score": round(citation.score, 4),
            }
            for citation in answer.citations
        ],
    }


def groundedness_score(answer_text: str, citations: list[SearchResult]) -> float:
    """Estimate how much of the answer vocabulary is supported by retrieved context."""
    answer_terms = set(tokenize(answer_text))
    if not answer_terms:
        return 0.0
    context_terms = set()
    for citation in citations:
        context_terms.update(tokenize(citation.text))
    return round(len(answer_terms & context_terms) / len(answer_terms), 3)


def format_answer(answer: Answer) -> str:
    lines = [f"Question: {answer.question}", "", f"Answer: {answer.answer}", ""]
    lines.append(f"Groundedness score: {answer.groundedness_score:.3f}")
    lines.append("")
    lines.append("Top citations:")
    for item in answer.citations:
        lines.append(f"- {item.source} / {item.chunk_id} | score={item.score:.3f}")
    return "\n".join(lines)
