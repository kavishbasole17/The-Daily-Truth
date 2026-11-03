FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch first so requirements.txt doesn't pull the multi-GB CUDA wheel.
# The GPU override (docker-compose.gpu.yml) rebuilds with TORCH_INDEX=cu121.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m app && mkdir -p /data /app/cache && chown -R app /data /app/cache
USER app

ENV HF_HOME=/home/app/.cache/huggingface \
    DATABASE_URL=sqlite:////data/paper_trail.db

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
