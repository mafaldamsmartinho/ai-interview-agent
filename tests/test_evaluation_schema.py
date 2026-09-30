from src.agent.state import Evaluation


def test_evaluation_schema():
    evaluation = Evaluation(
        correctness="Correct",
        clarity="Clear",
        missing_concepts="None significant",
        improved_answer="Good answer",
        verdict="good",
        confidence="high",
    )

    assert evaluation.verdict == "good"
