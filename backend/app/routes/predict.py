import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.models.schemas import PredictResponse
from backend.app.services.predict_service import run_prediction

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Unsupported image type.")

    try:
        contents = await file.read()
        result = run_prediction(contents)
        return PredictResponse(
            label=result.label,
            class_index=result.class_index,
            probabilities=result.probabilities,
            latency_ms=result.latency_ms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.exception("artifact_missing")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("prediction_failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
