import pytest
from pydantic import ValidationError

from schemas import Evaluation


@pytest.mark.parametrize("field", ["verdict", "confidence"])
def test_evaluation_rejects_invalid_labels(field):
    values = {
        "correctness": "Good",
        "clarity": "Good",
        "missing_concepts": "None",
        "improved_answer": "",
        "verdict": "good",
        "confidence": "high",
    }
    values[field] = "banana"

    with pytest.raises(ValidationError) as error:
        Evaluation(**values)

    assert error.value.errors()[0]["loc"] == (field,)
