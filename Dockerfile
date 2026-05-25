FROM python:3.11-slim-bookworm

WORKDIR /app

# Merge all system + Python deps into one RUN layer to reduce image size and vulnerabilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
# Install CPU-only PyTorch first — avoids pulling the 2 GB CUDA build — then the rest
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model into the image (no runtime download needed)
COPY scripts/download_model.py scripts/download_model.py
RUN HF_HOME=/app/.cache python scripts/download_model.py

# Copy the rest of the project
COPY . .

ENV HF_HOME=/app/.cache
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV PYTHONUNBUFFERED=1

# Use a start script so exec form CMD works with $PORT
COPY start.sh .
RUN chmod +x start.sh

EXPOSE ${PORT:-10000}

CMD ["./start.sh"]
