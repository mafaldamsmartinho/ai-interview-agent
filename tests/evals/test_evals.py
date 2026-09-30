import pytest

from schemas import Evaluation, InterviewPlan
from src.models.models import ModelProvider, get_model
from src.prompts.prompts import evaluation_prompt, planner_prompt
from tests.evals.evaluator_cases import EVALUATOR_CASES
from tests.evals.planner_cases import PLANNER_CASES

pytestmark = pytest.mark.skip(reason="LLM evals are non-deterministic and run manually")


def run_evaluator_evals():
    llm = get_model(ModelProvider.STRONG)
    chain = evaluation_prompt | llm.with_structured_output(Evaluation).with_retry(
        stop_after_attempt=2
    )

    passed = 0

    print("\n--- EVALUATOR EVALS ---")

    for i, case in enumerate(EVALUATOR_CASES, start=1):
        result = chain.invoke(
            {
                "question": case["question"],
                "answer": case["answer"],
            }
        )

        success = result.verdict == case["expected"]
        passed += success

        print(
            f"Case {i}: "
            f"{'PASS' if success else 'FAIL'} | "
            f"expected={case['expected']} | "
            f"actual={result.verdict}"
        )

    print(f"\nEvaluator: {passed}/{len(EVALUATOR_CASES)}")

    return passed


def run_planner_evals():
    llm = get_model(ModelProvider.FAST)
    chain = planner_prompt | llm.with_structured_output(InterviewPlan).with_retry(
        stop_after_attempt=2
    )

    passed = 0

    print("\n--- PLANNER EVALS ---")

    for i, case in enumerate(PLANNER_CASES, start=1):
        evaluation = Evaluation(
            correctness="Test evaluation",
            clarity="Test clarity",
            missing_concepts=case["missing_concepts"],
            improved_answer="Test improved answer",
            verdict=case["verdict"],
            confidence=case["confidence"],
        )

        result = chain.invoke(
            {
                "topic": "Machine Learning",
                "skill": "Overfitting",
                "question": "Explain overfitting.",
                "evaluation": evaluation.model_dump(),
                "memory": [],
            }
        )

        success = result.action in case["allowed_actions"]
        passed += success

        print(
            f"Case {i}: "
            f"{'PASS' if success else 'FAIL'} | "
            f"allowed={case['allowed_actions']} | "
            f"actual={result.action}"
        )

    print(f"\nPlanner: {passed}/{len(PLANNER_CASES)}")

    return passed


def test_agent_evals():
    evaluator_passed = run_evaluator_evals()
    planner_passed = run_planner_evals()

    assert evaluator_passed >= 4
    assert planner_passed >= 4
