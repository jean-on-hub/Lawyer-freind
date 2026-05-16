FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by pymupdf and faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model into the image (no runtime download needed)
RUN HF_HOME=/app/.cache python scripts/download_model.py 2>/dev/null || \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the project
COPY . .

ENV HF_HOME=/app/.cache
ENV PYTHONUNBUFFERED=1

EXPOSE 10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120"]
