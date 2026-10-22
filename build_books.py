"""Build data/books.json from the Jamalon Arabic Books dataset (Kaggle).

Source: https://www.kaggle.com/datasets/dareenalharthi/jamalon-arabic-books-dataset
(~9,000 Arabic books scraped from the Jamalon bookstore by Dareen Alharthi.)

Steps:
  1. read the CSV
  2. clean the text (Unicode NFKC fixes Arabic "presentation form" letters)
  3. keep books that have a real description, drop duplicate titles
  4. pick N books balanced across categories (fixed seed = same books every run)

Usage:  python build_books.py --csv "jamalon dataset.csv" --limit 200
"""
import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).parent / "data" / "books.json"
MISSING = {"", "none", "null", "nan", "لا يوجد"}


def clean(text: str | None) -> str:
    # NFKC turns presentation forms like "ﻣﺬﻛﺮات" into normal letters "مذكرات"
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.lower() in MISSING else text


def load_books(csv_path: Path) -> list[dict]:
    books, seen_titles = [], set()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for n, row in enumerate(csv.DictReader(f)):
            title = clean(row.get("Title"))
            description = clean(row.get("Description"))
            if not title or len(description) < 100 or title in seen_titles:
                continue
            seen_titles.add(title)
            books.append({
                "id": f"jm-{n}",
                "title": title,
                "author": clean(row.get("Author")),
                "category": clean(row.get("Category")),
                "subcategory": clean(row.get("Subcategory")),
                "year": "" if clean(row.get("Publication year")) in {"", "0"} else clean(row.get("Publication year")),
                "publisher": clean(row.get("Publisher")),
                "description": description[:800],
            })
    return books


def balanced_sample(books: list[dict], limit: int, seed: int) -> list[dict]:
    """Round-robin over categories so no single genre dominates."""
    rng = random.Random(seed)
    by_category = defaultdict(list)
    for b in books:
        by_category[b["category"] or "أخرى"].append(b)
    for group in by_category.values():
        rng.shuffle(group)
    picked = []
    while len(picked) < limit and any(by_category.values()):
        for category in sorted(by_category):
            if by_category[category] and len(picked) < limit:
                picked.append(by_category[category].pop())
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="jamalon dataset.csv")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    books = load_books(Path(args.csv))
    print(f"{len(books)} books have a real description (after removing duplicates)")

    books = balanced_sample(books, args.limit, args.seed)
    print(f"Picked {len(books)} books, balanced across categories:")
    for category, count in Counter(b["category"] for b in books).most_common():
        print(f"  {count:3d}  {category}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(books, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
