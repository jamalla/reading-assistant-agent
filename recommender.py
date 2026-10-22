"""Recommender: a simple search over ~200 Arabic book descriptions.

No model is trained. We turn every book into a TF-IDF vector of character
n-grams, do the same for the user's query, and return the closest books.
"""
import json
import re
import unicodedata
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "data" / "books.json"

# Arabic diacritics (tashkeel: fatha, damma, kasra, shadda, sukun...) + tatweel
_TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")


def normalize_arabic(text: str) -> str:
    """Make different spellings of the same word look the same."""
    text = unicodedata.normalize("NFKC", text)  # presentation forms -> normal letters
    text = _TASHKEEL.sub("", text)
    text = re.sub("[إأآٱ]", "ا", text)  # all alef forms -> ا
    text = text.replace("ة", "ه")       # taa marbuta -> haa
    text = text.replace("ى", "ي")       # alef maqsura -> yaa
    text = text.translate(str.maketrans("ھیک", "هيك"))  # Persian look-alikes -> Arabic
    text = re.sub(r"[^\w\s]", " ", text)  # drop punctuation like « » ، ؟
    return re.sub(r"\s+", " ", text).strip().lower()


class Recommender:
    def __init__(self, path: Path = DATA_PATH):
        self.books = json.loads(Path(path).read_text(encoding="utf-8"))
        # Title is repeated so it counts a bit more than the long description.
        docs = [
            normalize_arabic(
                f"{b['title']} {b['title']} {b.get('category', '')} {b.get('subcategory', '')} "
                f"{b.get('author', '')} {b['description']}"
            )
            for b in self.books
        ]
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(docs)
        self.ids = {b["id"] for b in self.books}
        self.titles = {normalize_arabic(b["title"]) for b in self.books}

    def search(self, query: str, k: int = 5, min_score: float = 0.05) -> list[dict]:
        """Return up to k books most similar to the query."""
        q = self.vectorizer.transform([normalize_arabic(query)])
        scores = cosine_similarity(q, self.matrix).ravel()
        results = []
        for i in scores.argsort()[::-1][:k]:
            if scores[i] < min_score:
                break
            b = self.books[i]
            results.append({
                "id": b["id"],
                "title": b["title"],
                "author": b.get("author", ""),
                "category": b.get("category", ""),
                "subcategory": b.get("subcategory", ""),
                "description": b["description"][:300],
                "score": round(float(scores[i]), 3),
            })
        return results


if __name__ == "__main__":
    import sys

    rec = Recommender()
    query = " ".join(sys.argv[1:]) or "رواية تاريخية قصيرة"
    print(f"Query: {query}\n")
    for r in rec.search(query):
        print(f"{r['score']:.3f}  {r['title']}  —  {r['author']}  [{r['category']} / {r['subcategory']}]")
