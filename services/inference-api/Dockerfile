FROM python:3.12-slim

WORKDIR /app

# System dependencies (Pillow + OpenCV runtime for ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch (smaller image, separate layer for caching)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1 torchvision==0.19.1

# Other deps
COPY services/inference-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model at build time (cached layer, no runtime download)
RUN mkdir -p /app/models && \
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && \
    mv yolov8n.pt /app/models/yolov8n.pt

ENV MODEL_PATH=/app/models/yolov8n.pt

# App code (last layer, changes most often)
COPY services/inference-api/main.py .
COPY services/inference-api/model.py .

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
