# مساعد القراءة — Reading Assistant Agent

An Arabic chat agent that recommends real books, evaluated with Langfuse and deployed on AWS.

1. **Recommender**: TF-IDF search over ~200 Arabic book descriptions (`recommender.py`)
2. **Agent**: LangChain agent on Amazon Bedrock that calls the recommender as a tool
3. **Evals**: Langfuse traces, a test dataset, scores, and prompt v1 vs v2 comparison
4. **Deployment**: one container on Amazon ECS Express Mode with HTTPS

## Build the book catalog

Download `jamalon dataset.csv` from the Kaggle dataset below, put it in the repo folder, then:

```bash
pip install -r requirements.txt
python build_books.py --csv "jamalon dataset.csv" --limit 200   # writes data/books.json
python recommender.py "رواية تاريخية قصيرة"
```

## Data attribution

Book titles, authors, categories and descriptions come from the
[Jamalon Arabic Books Dataset](https://www.kaggle.com/datasets/dareenalharthi/jamalon-arabic-books-dataset)
by Dareen Alharthi on Kaggle, collected from the [Jamalon](https://jamalon.com) online bookstore.
This project uses a 200-book sample of catalog metadata for educational purposes.
