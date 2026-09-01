# Truth Shield — single image serving the Svelte UI and the FastAPI API.
#
# Sized for a CPU-only host (Hugging Face Spaces free tier, Render, Fly, any
# Docker PaaS). Model weights are baked in at build time so cold starts do not
# pull half a gigabyte over the network.

# ── stage 1: build the Svelte bundle ──────────────────────────────────────────
FROM node:22-slim AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# Same-origin by default; override to point the UI at a separately hosted API.
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build


# ── stage 2: python runtime ───────────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000; matching it keeps the caches
# below writable there and everywhere else.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface \
    NLTK_DATA=/home/user/nltk_data \
    PORT=7860

WORKDIR $HOME/app

COPY --chown=user:user backend/requirements.txt ./

# CPU-only torch first: the default PyPI wheel drags in ~2.5 GB of CUDA
# libraries this image can never use. Installing it up front means the
# requirements pass sees `torch` already satisfied.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user backend/ ./

# Bake in everything the first request would otherwise have to fetch.
RUN python download_models.py \
 && python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')" \
 && python -c "from transformers import GPT2TokenizerFast, GPT2LMHeadModel; \
GPT2TokenizerFast.from_pretrained('gpt2'); GPT2LMHeadModel.from_pretrained('gpt2')"

COPY --from=frontend --chown=user:user /build/dist ./frontend_dist

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT:-7860}/health" || exit 1

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
