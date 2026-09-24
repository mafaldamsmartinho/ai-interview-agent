from state import Evaluation


def test_evaluation_schema():
    evaluation = Evaluation(
        correctness="Correct",
        clarity="Clear",
        missing_concepts="None significant",
        improved_answer="Good answer",
        verdict="good",
    )

    assert evaluation.verdict == "good"
