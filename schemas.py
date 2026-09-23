from pydantic import BaseModel, Field


class Evaluation(BaseModel):
    correctness: str
    clarity: str
    missing_concepts: str
    improved_answer: str
    score: int = Field(ge=0, le=20)
