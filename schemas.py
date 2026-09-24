from typing import Literal

from pydantic import BaseModel

from models import ModelProvider

VERDICT_TO_SCORE = {
    "excellent": 20,
    "good": 16,
    "partial": 10,
    "poor": 5,
}


class Evaluation(BaseModel):
    correctness: str
    clarity: str
    missing_concepts: str
    improved_answer: str
    verdict: Literal["excellent", "good", "partial", "poor"]


class RoutingDecision(BaseModel):
    model: ModelProvider
    reason: str
