"""Upload evals/questions.json to Langfuse as a dataset (safe to re-run).

Each item keeps a stable id, so running this again updates items instead of
creating duplicates.

Usage:  python evals/upload_dataset.py
"""
import json
from pathlib import Path

from langfuse import get_client

DATASET = "reading-assistant-evals"
QUESTIONS = Path(__file__).parent / "questions.json"


def main() -> None:
    langfuse = get_client()  # reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / BASE_URL
    langfuse.create_dataset(
        name=DATASET,
        description="12 Arabic book requests: 8 normal + 4 traps (out of catalog, coverage gaps).",
    )
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    for q in questions:
        langfuse.create_dataset_item(
            dataset_name=DATASET,
            id=f"{DATASET}-{q['id']}",
            input={"question": q["question"]},
            expected_output=q["expected"],
            metadata={"qid": q["id"], "type": q["type"]},
        )
        print(f"  ✓ {q['id']}  [{q['type']}]  {q['question']}")
    langfuse.flush()
    print(f"\nDataset '{DATASET}' has {len(questions)} items.")


if __name__ == "__main__":
    main()
