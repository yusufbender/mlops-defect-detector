from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from PIL import Image
import io
import time
import random
import uuid

app = FastAPI(
    title="Defect Detection API",
    description="Industrial defect detection inference service",
    version="0.1.0"
)

# ----- Prometheus metrics -----
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests",
    ["endpoint", "status"]
)
INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Inference duration in seconds",
    ["endpoint"]
)
ACTIVE_REQUESTS = Gauge(
    "inference_active_requests",
    "Currently active inference requests"
)
DEFECTS_DETECTED = Counter(
    "defects_detected_total",
    "Total defects detected",
    ["defect_type"]
)

# ----- Mock defect classes (MVTec AD inspired) -----
DEFECT_TYPES = ["scratch", "dent", "crack", "stain", "hole", "ok"]


@app.get("/")
async def root():
    return {
        "service": "defect-detection-api",
        "version": "0.1.0",
        "status": "ok"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    # In future: check model loaded, storage available, etc.
    return {"status": "ready"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image, returns mock defect detection result.
    Real YOLOv8 model integration comes in Phase 2.
    """
    ACTIVE_REQUESTS.inc()
    start = time.time()
    request_id = str(uuid.uuid4())[:8]

    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith("image/"):
            REQUEST_COUNT.labels("/predict", "400").inc()
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and validate image
        contents = await file.read()
        try:
            img = Image.open(io.BytesIO(contents))
            img.verify()
            img = Image.open(io.BytesIO(contents))  # reopen after verify
            width, height = img.size
        except Exception:
            REQUEST_COUNT.labels("/predict", "400").inc()
            raise HTTPException(status_code=400, detail="Invalid image file")

        # ----- MOCK INFERENCE -----
        # Simulate model processing time
        time.sleep(random.uniform(0.05, 0.2))

        # Random defect detection result
        is_defective = random.random() > 0.4
        if is_defective:
            defect_type = random.choice(DEFECT_TYPES[:-1])  # exclude "ok"
            confidence = round(random.uniform(0.75, 0.99), 3)
            DEFECTS_DETECTED.labels(defect_type).inc()
            bbox = [
                random.randint(0, width // 2),
                random.randint(0, height // 2),
                random.randint(width // 2, width),
                random.randint(height // 2, height),
            ]
        else:
            defect_type = "ok"
            confidence = round(random.uniform(0.85, 0.99), 3)
            bbox = None

        REQUEST_COUNT.labels("/predict", "200").inc()

        return {
            "request_id": request_id,
            "filename": file.filename,
            "image_size": {"width": width, "height": height},
            "defective": is_defective,
            "defect_type": defect_type,
            "confidence": confidence,
            "bbox": bbox,
            "model": "mock-v0",
            "inference_time_ms": int((time.time() - start) * 1000),
        }
    finally:
        INFERENCE_LATENCY.labels("/predict").observe(time.time() - start)
        ACTIVE_REQUESTS.dec()


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
