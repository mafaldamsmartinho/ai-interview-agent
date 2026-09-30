from typing import NotRequired, TypedDict

from schemas import Evaluation, InterviewPlan


class InterviewState(TypedDict):
    topic: str
    skill: str
    question: str
    previous_question: str
    answer: str
    evaluation: Evaluation | None
    continue_interview: bool
    interview_plan: NotRequired[InterviewPlan]
