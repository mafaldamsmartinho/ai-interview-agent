from typing import Any, Literal

from pydantic import BaseModel, Field

from src.models.models import ModelProvider

VERDICT_TO_SCORE = {
    "excellent": 10,
    "good": 7,
    "partial": 4,
    "poor": 2,
}


class Evaluation(BaseModel):
    correctness: str
    clarity: str
    missing_concepts: str
    improved_answer: str
    verdict: Literal["excellent", "good", "partial", "poor"]
    confidence: Literal["high", "medium", "low"]


class RoutingDecision(BaseModel):
    model: ModelProvider
    reason: str


class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class InterviewQuestion(BaseModel):
    question: str
    skill: str


class InterviewPlan(BaseModel):
    action: Literal[
        "probe",
        "harder",
        "easier",
        "new_skill",
        "coach",
    ]
    reason: str


class CoachOutput(BaseModel):
    explanation: str
    study_recommendation: str
