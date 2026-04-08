FROM python:3.10-slim

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy and install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY server/ ./server/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Expose Hugging Face default port
EXPOSE 7860

# Health-check so HF knows when the container is ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
