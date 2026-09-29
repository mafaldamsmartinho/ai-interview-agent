from src.agent.nodes import route_by_score
from src.agent.state import Evaluation


def test_high_score_routes_to_harder():
    state = {
        "evaluation": Evaluation(
            correctness="Good",
            clarity="Good",
            missing_concepts="None",
            improved_answer="",
            verdict="good",
        )
    }

    assert route_by_score(state) == "same"
