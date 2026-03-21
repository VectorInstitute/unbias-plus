# ============================================================
# UnBias Plus Demo — CPU Cloud Run Service
# Base: Python 3.11 slim (no CUDA — inference is offloaded to
#       the vLLM GPU service via VLLM_BASE_URL).
# Model: served remotely by unbias-plus-vllm (Cloud Run GPU)
# ============================================================
FROM python:3.11-slim

LABEL maintainer="Vector Institute AI Engineering"
LABEL description="UnBias Plus — bias detection and debiasing demo (proxy to vLLM)"

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy project files (pyproject.toml first for layer caching)
COPY pyproject.toml README.md ./
COPY src/ src/

# Install CPU-only torch to satisfy the package's torch dependency without
# pulling in multi-GB CUDA wheels. When VLLM_BASE_URL is set (all Cloud Run
# deployments), torch is never actually imported at runtime.
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip cache purge

# Install unbias-plus and remaining deps (torch already satisfied above).
# openai is added for the vLLM OpenAI-compatible client.
RUN pip install --no-cache-dir \
    openai \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -e "." \
    && pip cache purge

ENV HF_HOME=/app/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false
ENV TQDM_DISABLE=1

# Cloud Run injects $PORT (default 8080)
ENV PORT=8080
EXPOSE 8080

COPY nginx.conf entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh


# Run as non-root for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# nginx serves the UI on port 8080 immediately; uvicorn loads in the background.
CMD ["/app/entrypoint.sh"]
