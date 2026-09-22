from typing import TypedDict

from pydantic import BaseModel


class Evaluation(BaseModel):
    correctness: str
    clarity: str
    missing_concepts: str
    improved_answer: str
    score: int


class InterviewState(TypedDict):
    topic: str
    question: str
    previous_question: str
    answer: str
    evaluation: Evaluation | None
    continue_interview: bool
