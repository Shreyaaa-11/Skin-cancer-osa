from typing import List

from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    label: str
    class_index: int
    probabilities: List[float] = Field(..., min_length=7, max_length=7)
    latency_ms: float
