# BERTurk MCP server (stdio). CPU-only image; mount a checkpoint directory at /models.
#   docker build -t tcls-mcp .
#   docker run -i --rm -v /abs/path/to/checkpoint:/models:ro tcls-mcp
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

WORKDIR /app

# CPU wheel of torch first (~200 MB instead of ~2.5 GB of CUDA libraries).
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

RUN useradd --create-home --uid 1000 app
USER app

ENV MODEL_PATH=/models \
    DEVICE=cpu \
    MAX_BATCH=64

# stdio transport: the client must keep stdin open (docker run -i).
ENTRYPOINT ["python", "-m", "tcls_mcp"]
