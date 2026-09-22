import pytest
from pydantic import ValidationError

from state import Evaluation


def test_score_cannot_be_invalid():
    with pytest.raises(ValidationError):
        Evaluation(
            correctness="Good",
            clarity="Good",
            missing_concepts="None",
            improved_answer="",
            score="banana",
        )
