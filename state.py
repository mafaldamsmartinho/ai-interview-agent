from typing import TypedDict

from schemas import Evaluation


class InterviewState(TypedDict):
    topic: str
    question: str
    previous_question: str
    answer: str
    evaluation: Evaluation | None
    continue_interview: bool
