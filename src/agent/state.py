from typing import NotRequired, TypedDict

from schemas import Evaluation


class InterviewState(TypedDict):
    topic: str
    question: str
    previous_question: str
    answer: str
    evaluation: Evaluation | None
    tool_result: NotRequired[dict[str, str]]
    continue_interview: bool
