import argparse
import json

from rag_engine import answer_question


def main():
    parser = argparse.ArgumentParser(description="Run the local context-aware RAG demo.")
    parser.add_argument("question", help="Question to ask the local document collection.")
    parser.add_argument("--doc-dir", default="sample_docs", help="Directory containing .txt documents.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    result = answer_question(args.question, doc_dir=args.doc_dir, top_k=args.top_k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
