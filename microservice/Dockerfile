ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

FROM base AS builder

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM base AS dev

COPY --from=builder /install /usr/local

RUN pip install watchfiles

EXPOSE 8000

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload"]

FROM base AS test

COPY --from=builder /install /usr/local

COPY tests/requirements-test.txt ./tests/requirements-test.txt
RUN pip install --no-cache-dir -r tests/requirements-test.txt

COPY app/   app/
COPY tests/ tests/

ENV PYTHONPATH=/app

CMD ["pytest", "tests/unit/", "-m", "unit", "--tb=short", "-q"]

FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/appuser \
    PYTHONUSERBASE=/home/appuser/.local \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app/ app/

RUN useradd --create-home --shell /bin/false --uid 1001 appuser \
    && mkdir -p /home/appuser/.local \
    && chown -R appuser:appuser /home/appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/')" \
    || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
