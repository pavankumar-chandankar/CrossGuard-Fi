"""FastAPI serving with opt-in shared-secret authentication and readiness semantics."""
import hmac
import os
from pathlib import Path
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from crossguard_fi.config import CONTRACT
from crossguard_fi.serving.predictor import Predictor


class WindowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    samples: list[list[float]]
    timestamps_ns: list[int]
    top_k: int = Field(default=3, ge=1, le=CONTRACT.classes)

    @field_validator("samples")
    @classmethod
    def check_shape(cls, samples):
        if len(samples) != CONTRACT.window_samples:
            raise ValueError("Exactly 100 time samples required")
        if any(len(row) != CONTRACT.channels for row in samples):
            raise ValueError("Each sample must contain exactly 270 channels")
        return samples

    @field_validator("timestamps_ns")
    @classmethod
    def check_timestamps(cls, timestamps):
        if len(timestamps) != CONTRACT.window_samples:
            raise ValueError("Exactly 100 timestamps required")
        return timestamps


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    windows: list[WindowRequest] = Field(min_length=1, max_length=8)


def create_app(predictor: Predictor | None = None) -> FastAPI:
    app = FastAPI(title="CrossGuard-Fi API", version="0.2.0", docs_url=None,
                  redoc_url=None, openapi_url=None)
    if predictor is None and os.getenv("CGF_MODEL_PATH"):
        predictor = Predictor(Path(os.environ["CGF_MODEL_PATH"]))
    app.state.predictor = predictor
    api_key = os.getenv("CGF_API_KEY", "")
    if os.getenv("CGF_ENV", "development") == "production" and not api_key:
        raise RuntimeError("Production requires CGF_API_KEY from a secret store")

    @app.middleware("http")
    async def request_cap(request: Request, call_next):
        # Defense in depth; front proxy should also impose size/rate limits.
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > 3_000_000:
                    return JSONResponse({"detail": "Request body too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid content length"}, status_code=400)
        return await call_next(request)

    def require_key(x_api_key: str | None = Header(default=None)):
        if api_key and not hmac.compare_digest(x_api_key or "", api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def require_model() -> Predictor:
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="No validated model checkpoint loaded")
        return app.state.predictor

    @app.get("/healthz")
    def health():
        return {"status": "alive", "service": "crossguard-fi"}

    @app.get("/readyz")
    def ready():
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        return {"status": "ready", "model_sha256": app.state.predictor.model_sha256}

    @app.get("/v1/model", dependencies=[Depends(require_key)])
    def info():
        model = require_model()
        return {"model_sha256": model.model_sha256,
                "model_family": model.metadata["model"], "contract": CONTRACT.to_dict(),
                "label_names": model.metadata["label_names"]}

    @app.post("/v1/predict", dependencies=[Depends(require_key)])
    def predict(payload: WindowRequest):
        model = require_model()
        started = time.perf_counter()
        try:
            result = model.predict(payload.samples, payload.timestamps_ns, payload.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    @app.post("/v1/predict-batch", dependencies=[Depends(require_key)])
    def predict_batch(payload: BatchRequest):
        model = require_model()
        results = []
        for window in payload.windows:
            try:
                results.append(model.predict(window.samples, window.timestamps_ns,
                                             window.top_k))
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"predictions": results, "count": len(results)}

    return app


app = create_app()
