# Cognifin API. Build context: backend/
#   docker compose build            (from the repo root)
# The index is NOT baked in; the container fetches it into /data at first boot
# (ASSET_MODE=remote, HF_CACHE_URL). Model weights ARE baked in, so boot never
# waits on Hugging Face.

FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /venv
ENV PATH=/venv/bin:$PATH

# CPU-only torch first: the default wheel drags in ~2 GB of CUDA libraries.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Must match EMBEDDING_MODEL / RERANKER_MODEL at runtime (HF_HUB_OFFLINE below).
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
ARG RERANKER_MODEL=BAAI/bge-reranker-base
ENV HF_HOME=/models
RUN python -c "import sys; from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer(sys.argv[1]); CrossEncoder(sys.argv[2])" "$EMBEDDING_MODEL" "$RERANKER_MODEL"


FROM python:3.12-slim
ENV PATH=/venv/bin:$PATH HF_HOME=/models HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
RUN useradd --create-home app && mkdir /data && chown app /data
COPY --from=build /venv /venv
# Writable by app: HF libraries may create lock files next to cached weights
COPY --from=build --chown=app /models /models
WORKDIR /app
COPY . .
USER app

EXPOSE 8000
# Only Caddy can reach this port (compose publishes nothing else), so trusting
# its X-Forwarded-For is safe — and required, or every visitor shares one
# rate-limit bucket keyed on Caddy's IP.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
