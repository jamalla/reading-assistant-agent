"""Scoring rules for the reading assistant (all scores are 0..1).

real_books   - share of recommended titles that really exist in data/books.json.
               An LLM only *lists* the titles; plain code decides if each exists.
fits_request - LLM judge: does the answer fit the request and the expected
               behaviour, given the books the tool actually returned?
"""
import json
import os
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI
from langfuse import Evaluation
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from recommender import DATA_PATH, normalize_arabic  # noqa: E402

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4-2026-03-05")  # pinned, stronger than the agent
CATALOG = [normalize_arabic(b["title"]) for b in json.loads(DATA_PATH.read_text(encoding="utf-8"))]


class RecommendedTitles(BaseModel):
    titles: list[str] = Field(
        description="Exact titles of books the assistant RECOMMENDS to the reader. "
        "Exclude books it says are unavailable or not in the catalog."
    )


class FitVerdict(BaseModel):
    score: int = Field(ge=1, le=5, description="1 = wrong or misleading, 5 = perfect fit and honest")
    reason: str = Field(description="One short sentence in English")


def _judge():
    return ChatOpenAI(model=JUDGE_MODEL)


def in_catalog(title: str) -> bool:
    t = normalize_arabic(title)
    return any(t == c or (len(t) >= 6 and (t in c or c in t)) for c in CATALOG)


def real_books(*, input, output, expected_output=None, metadata=None, **kwargs):
    extract = _judge().with_structured_output(RecommendedTitles).invoke(
        "List the book titles recommended in this Arabic answer.\n\n" + output["answer"]
    )
    titles = extract.titles
    if not titles:
        return Evaluation(name="real_books", value=1.0, comment="No books recommended.")
    missing = [t for t in titles if not in_catalog(t)]
    value = (len(titles) - len(missing)) / len(titles)
    comment = f"Not in catalog: {missing}" if missing else f"All {len(titles)} titles are in the catalog."
    return Evaluation(name="real_books", value=round(value, 3), comment=comment)


def fits_request(*, input, output, expected_output=None, metadata=None, **kwargs):
    prompt = f"""You are grading an Arabic book-recommendation assistant.

Reader's request: {input["question"]}
Expected behaviour: {expected_output}
Books the search tool returned (the ONLY books in our catalog for this request): {output["retrieved"]}

Assistant's answer:
{output["answer"]}

Score 1-5: does the answer fit the request AND follow the expected behaviour?
Be strict: recommending books that are not in the returned list, or presenting
unrelated books as if they match, must score 1 or 2."""
    verdict = _judge().with_structured_output(FitVerdict).invoke(prompt)
    return Evaluation(name="fits_request", value=(verdict.score - 1) / 4, comment=verdict.reason)


def averages(*, item_results, **kwargs):
    """Run-level summary: the mean of each score across all questions."""
    totals = {}
    for result in item_results:
        for ev in result.evaluations:
            totals.setdefault(ev.name, []).append(ev.value)
    return [
        Evaluation(name=f"avg_{name}", value=round(sum(v) / len(v), 3), comment=f"{len(v)} items")
        for name, v in totals.items()
    ]
