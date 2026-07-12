import csv
import json
from pathlib import Path

from rag_engine import answer_question, build_chunks, load_documents, retrieve, tokenize


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_questions(path):
    with project_path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reciprocal_rank(sources, expected_source):
    for index, source in enumerate(sources, start=1):
        if source == expected_source:
            return 1 / index
    return 0


def term_recall(answer, expected_terms):
    expected = [term.strip().lower() for term in expected_terms.split("|") if term.strip()]
    if not expected:
        return 0
    answer_tokens = set(tokenize(answer))
    answer_text = answer.lower()
    hits = 0
    for term in expected:
        term_tokens = tokenize(term)
        if term in answer_text or all(token in answer_tokens for token in term_tokens):
            hits += 1
    return hits / len(expected)


def groundedness(answer, retrieved_text):
    answer_tokens = set(tokenize(answer))
    context_tokens = set(tokenize(retrieved_text))
    if not answer_tokens:
        return 0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def evaluate(
    questions_path="eval_data/questions.csv",
    doc_dir="sample_docs",
    output_path="outputs/eval_results.json",
    top_k=3,
):
    documents = load_documents(doc_dir)
    chunks = build_chunks(documents)
    rows = read_questions(questions_path)
    detailed = []

    hit_total = 0
    reciprocal_total = 0
    recall_total = 0
    grounded_total = 0

    for row in rows:
        question = row["question"]
        expected_source = row["expected_source"]
        expected_terms = row.get("expected_terms", "")
        retrieved = retrieve(question, chunks, top_k=top_k)
        result = answer_question(question, doc_dir=doc_dir, top_k=top_k)
        retrieved_sources = [chunk.source for _, chunk in retrieved]
        retrieved_text = " ".join(chunk.text for _, chunk in retrieved)

        hit = int(expected_source in retrieved_sources)
        rr = reciprocal_rank(retrieved_sources, expected_source)
        recall = term_recall(result["answer"], expected_terms)
        ground = groundedness(result["answer"], retrieved_text)

        hit_total += hit
        reciprocal_total += rr
        recall_total += recall
        grounded_total += ground

        detailed.append(
            {
                "question": question,
                "expected_source": expected_source,
                "retrieved_sources": retrieved_sources,
                "hit": hit,
                "reciprocal_rank": round(rr, 4),
                "answer_term_recall": round(recall, 4),
                "groundedness": round(ground, 4),
                "answer": result["answer"],
            }
        )

    n = len(rows) or 1
    summary = {
        "hit_at_k": round(hit_total / n, 4),
        "mean_reciprocal_rank": round(reciprocal_total / n, 4),
        "mean_answer_term_recall": round(recall_total / n, 4),
        "mean_groundedness": round(grounded_total / n, 4),
        "top_k": top_k,
        "num_questions": len(rows),
    }

    output = {"summary": summary, "details": detailed}
    output_file = project_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return output


if __name__ == "__main__":
    evaluate()
