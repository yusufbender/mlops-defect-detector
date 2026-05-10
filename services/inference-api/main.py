import io
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from PIL import Image

from model import detector

# ----- Logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("inference-api")

# ----- Prometheus metrics -----
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["endpoint", "status"],
)
INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Inference duration in seconds",
    ["endpoint"],
)
ACTIVE_REQUESTS = Gauge(
    "inference_active_requests",
    "Currently active inference requests",
)
DEFECTS_DETECTED = Counter(
    "defects_detected_total",
    "Total defects detected",
    ["defect_type"],
)
MODEL_LOAD_TIME = Gauge(
    "model_load_duration_seconds",
    "Time taken to load the model at startup",
)


# ----- Lifespan: load model on startup -----
@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.time()
    try:
        detector.load()
        elapsed = time.time() - start
        MODEL_LOAD_TIME.set(elapsed)
        logger.info("Model loaded in %.2fs", elapsed)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        raise
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Defect Detection API",
    description="Industrial defect detection inference service",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "service": "defect-detection-api",
        "version": "0.2.0",
        "model": detector.version,
        "status": "ok",
    }


@app.get("/health")
async def health():
    """Liveness probe: process is up."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness probe: model is loaded."""
    if not detector.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready", "model": detector.version}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ACTIVE_REQUESTS.inc()
    start = time.time()
    request_id = str(uuid.uuid4())[:8]

    try:
        if not detector.is_ready():
            REQUEST_COUNT.labels("/predict", "503").inc()
            raise HTTPException(status_code=503, detail="Model not ready")

        if not file.content_type or not file.content_type.startswith("image/"):
            REQUEST_COUNT.labels("/predict", "400").inc()
            raise HTTPException(status_code=400, detail="File must be an image")

        contents = await file.read()
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            width, height = img.size
        except Exception:
            REQUEST_COUNT.labels("/predict", "400").inc()
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Real inference
        result = detector.predict(img)

        # Update defect counter
        if result["defective"]:
            DEFECTS_DETECTED.labels(result["defect_type"]).inc()

        REQUEST_COUNT.labels("/predict", "200").inc()
        elapsed_ms = int((time.time() - start) * 1000)

        logger.info(
            "request_id=%s defective=%s type=%s conf=%.3f elapsed_ms=%d",
            request_id, result["defective"], result["defect_type"],
            result["confidence"], elapsed_ms,
        )

        return {
            "request_id": request_id,
            "filename": file.filename,
            "image_size": {"width": width, "height": height},
            "inference_time_ms": elapsed_ms,
            **result,
        }
    finally:
        INFERENCE_LATENCY.labels("/predict").observe(time.time() - start)
        ACTIVE_REQUESTS.dec()


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
