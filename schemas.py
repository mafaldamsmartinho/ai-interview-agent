from typing import Any, Literal

from pydantic import BaseModel, Field

from src.models.models import ModelProvider

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


class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolGuardDecision(BaseModel):
    decision: Literal["allow", "confirm", "deny"]
    reason: str
