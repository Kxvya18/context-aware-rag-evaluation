# Context-Aware RAG Assistant and Evaluation Toolkit

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Prototype%20Complete-brightgreen)
![Focus](https://img.shields.io/badge/Focus-GenAI%20%7C%20RAG%20%7C%20Evaluation-purple)

A local Retrieval-Augmented Generation (RAG) assistant that demonstrates the core AI engineering workflow behind grounded question answering: document ingestion, chunking, retrieval, answer construction, source citation, and evaluation.

This project is intentionally lightweight and explainable. It does not require paid APIs, hosted LLMs, vector databases, or external packages. The goal is to show the mechanics of a RAG pipeline clearly enough to explain in interviews, then leave a clean path for future extensions such as embeddings, FAISS/Chroma, reranking, LLM APIs, and a web interface.

## Project Snapshot

| Area | Details |
| --- | --- |
| Project type | Local GenAI / RAG / evaluation toolkit |
| Core idea | Retrieve relevant project context and generate grounded answers with citations |
| Retrieval method | TF-IDF-style lexical retrieval with cosine similarity |
| Knowledge base | Short technical documents from AI/ML research projects |
| Evaluation | Hit@K, Mean Reciprocal Rank, answer term recall, groundedness |
| Dependencies | Python standard library only |
| Output | JSON answers with sources and JSON evaluation results |

## Why This Project Matters

RAG systems are widely used in AI engineering because they reduce hallucination by grounding model responses in external context. However, a RAG application is only useful if retrieval quality, answer support, and citations can be measured.

This project focuses on those engineering fundamentals:

- preparing a small document knowledge base
- chunking documents into retrievable units
- ranking context for a user question
- generating answers from retrieved text
- returning citations for traceability
- evaluating retrieval and answer support with metrics
- producing reproducible JSON outputs

## Architecture

```text
User Question
     |
     v
Document Loader  -->  sample_docs/*.txt
     |
     v
Chunking With Overlap
     |
     v
TF-IDF Style Retrieval
     |
     v
Top Relevant Context Chunks
     |
     v
Grounded Answer Builder
     |
     v
Answer + Source Citations
     |
     v
Evaluation Harness
     |
     v
Hit@K, MRR, Term Recall, Groundedness
```

## Repository Structure

```text
context-aware-rag-evaluation/
  README.md
  requirements.txt
  .gitignore
  app.py
  rag_engine.py
  evaluate.py
  sample_docs/
    pirge.txt
    hypergraph_iiot.txt
    model_calibration.txt
    moga_fcm.txt
    eeg_topology.txt
  eval_data/
    questions.csv
  outputs/
    eval_results.json
```

## Files Explained

| File | Purpose |
| --- | --- |
| `app.py` | Command-line interface for asking questions over the local document collection |
| `rag_engine.py` | Core RAG logic: loading documents, chunking, TF-IDF retrieval, answer generation |
| `evaluate.py` | Evaluation harness for retrieval quality and answer support |
| `sample_docs/` | Small technical knowledge base used for local RAG demonstration |
| `eval_data/questions.csv` | Evaluation questions, expected sources, and expected answer terms |
| `outputs/eval_results.json` | Saved evaluation result artifact |
| `requirements.txt` | Empty dependency file because the current version uses only Python standard library |

## Sample Knowledge Base

The local knowledge base contains short public-style summaries of AI/ML project areas:

- PIRGE and PIRGE-W for pre-inference LLM resource prediction and water-aware execution control
- Hypergraph spectral analysis for coordinated IIoT attack detection
- Model calibration and overconfidence diagnostics
- MOGA-FCM for unsupervised cognitive load detection
- Topological data analysis for EEG functional connectivity

These documents are intentionally small so the complete retrieval pipeline can be inspected and explained.

## How To Run

Clone the repository:

```bash
git clone https://github.com/Kxvya18/context-aware-rag-evaluation.git
cd context-aware-rag-evaluation
```

Run a question:

```bash
python app.py "What does PIRGE predict before inference?"
```

Try more examples:

```bash
python app.py "Why are hypergraphs useful for coordinated IIoT attacks?"
python app.py "Which metrics are used for model calibration?"
python app.py "How does MOGA-FCM support cognitive load detection?"
python app.py "How is topology used for EEG functional connectivity?"
```

Run evaluation:

```bash
python evaluate.py
```

The evaluation output is written to:

```text
outputs/eval_results.json
```

## Example Output

Example command:

```bash
python app.py "Which metrics are used for model calibration?"
```

Example response:

```json
{
  "question": "Which metrics are used for model calibration?",
  "answer": "Model calibration studies whether predicted confidence matches actual correctness. A model can have high accuracy but still be overconfident, especially under noise or distribution shift. Calibration diagnostics include Expected Calibration Error, reliability diagrams, class-wise calibration, Brier score, confidence-error coupling, and temperature scaling.",
  "sources": [
    {
      "source": "model_calibration.txt",
      "chunk_index": 0,
      "score": 0.3217
    }
  ]
}
```

## Evaluation Metrics

| Metric | What It Measures |
| --- | --- |
| Hit@K | Whether the expected source appears in the retrieved top-k results |
| Mean Reciprocal Rank (MRR) | How highly the expected source is ranked |
| Answer Term Recall | Whether expected concepts appear in the generated answer |
| Groundedness | How much of the answer vocabulary is supported by retrieved context |

## Current Evaluation Result

On the included sample evaluation set:

```text
Hit@K: 1.0
MRR: 1.0
Mean Answer Term Recall: 1.0
Mean Groundedness: 1.0
Number of Questions: 5
```

## Design Decisions

### Why no external LLM API?

The first version avoids hosted APIs so the full pipeline is reproducible locally. This makes the retrieval and evaluation logic easier to inspect, debug, and explain.

### Why TF-IDF style retrieval?

TF-IDF is simple, transparent, and useful for demonstrating retrieval mechanics. It also provides a strong baseline before adding embeddings or vector databases.

### Why source citations?

Every generated answer includes source metadata so the user can trace where the answer came from. This is important for trustworthy GenAI systems.

### Why evaluation?

A RAG system should not only generate answers. It should also be tested for retrieval accuracy and answer support. The evaluation harness gives a small but useful framework for measuring this.

## AI Engineering Skills Demonstrated

- RAG pipeline design
- local document ingestion
- chunking strategy
- retrieval scoring
- grounded answer generation
- citation-aware output
- evaluation dataset creation
- metric design
- JSON artifact generation
- reproducible command-line workflows
- clean project documentation

## Limitations

- Uses lexical retrieval instead of dense embeddings.
- Uses short local documents rather than a large-scale corpus.
- Does not yet include reranking.
- Does not yet call an external LLM for abstractive generation.
- Does not yet include a web UI or API server.

These limitations are intentional for the first version because the goal is a transparent baseline that can be extended.

## Future Improvements

- Add sentence-transformer embeddings.
- Add FAISS or Chroma for vector search.
- Add reranking for better context selection.
- Add an LLM API for abstractive answer generation.
- Add hallucination checks and answer faithfulness scoring.
- Add query rewriting.
- Add Streamlit or FastAPI interface.
- Add MLflow experiment tracking.
- Add more documents and a larger evaluation set.

## Status

Prototype complete and portfolio-ready. The project is designed to be extended into a stronger AI engineering demo with embeddings, vector search, LLM integration, and a user interface.
