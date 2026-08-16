FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.0 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_DEV=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Copy dependency files first so Docker can cache this layer.
COPY pyproject.toml uv.lock ./

# Install the runtime dependencies declared in uv.lock.
RUN uv sync --locked --no-install-project

# Copy application source and fixed runtime artifacts.
COPY main.py ./
COPY apps ./apps
COPY clinical_synopsis ./clinical_synopsis
COPY models ./models
COPY data/retrieval ./data/retrieval
COPY data/derived ./data/derived

# Install the project itself.
RUN uv sync --locked

EXPOSE 8501

CMD ["streamlit", "run", "apps/Clinical_Synopsis.py", "--server.address=0.0.0.0", "--server.port=8501"]