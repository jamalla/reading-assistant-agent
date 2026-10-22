# Lesson 4: the container image that ECS runs.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the web app needs (no notebooks, no evals, no raw CSV).
COPY agent.py prompts.py recommender.py app.py ./
COPY data/books.json data/books.json

# Secrets are NOT baked in: ECS injects them as env vars at runtime.
ENV PROMPT_VERSION=v2 \
    PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
