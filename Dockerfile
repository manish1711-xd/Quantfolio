# syntax=docker/dockerfile:1
#
# Multi-stage build: wheels are compiled once in a builder stage and the runtime
# image installs only the prebuilt artefacts, so the shipped image carries no
# compiler toolchain and no build cache.

# ---------------------------------------------------------------------------
# stage 1 - build wheels
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip wheel --wheel-dir /wheels -r requirements.txt

# ---------------------------------------------------------------------------
# stage 2 - runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run as an unprivileged user; the container never needs root.
RUN useradd --create-home --uid 10001 quant

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY --chown=quant:quant . .

# SQLite portfolios live here; declare it so `docker run -v` persists them.
RUN mkdir -p /app/data && chown -R quant:quant /app/data
VOLUME ["/app/data"]

USER quant
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["streamlit", "run", "app.py"]
CMD ["--server.port=8501"]
