FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python >/dev/null 2>&1 || true \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless>=4.10.0

RUN useradd --create-home --shell /bin/sh aiuser

COPY --chown=aiuser:aiuser src ./src
COPY --chown=aiuser:aiuser models ./models

USER aiuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "activofijo_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
