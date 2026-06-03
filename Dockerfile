FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run as non-root user
RUN useradd -m botuser && chown -R botuser:botuser /app
USER botuser

ENV PYTHONPATH=/app
ENV DRY_RUN=true

EXPOSE 8501

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import json,os,time; s=json.load(open('data/state/bot-state.json')); exit(0 if time.time()-float(s.get('last_update',0))<180 else 1)" \
    || exit 1

CMD ["streamlit", "run", "scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
