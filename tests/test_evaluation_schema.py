from state import Evaluation


def test_evaluation_schema():
    evaluation = Evaluation(
        correctness="Correct",
        clarity="Clear",
        missing_concepts="None significant",
        improved_answer="Good answer",
        score=17,
    )

    assert evaluation.score == 17
