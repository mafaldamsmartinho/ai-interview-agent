from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import interrupt

from schemas import (
    VERDICT_TO_SCORE,
    CoachOutput,
    Evaluation,
    InterviewPlan,
    InterviewQuestion,
    ToolRequest,
)
from src.agent.state import InterviewState
from src.memory.repository import get_skill_profiles, update_skill_profile
from src.models.models import ModelProvider, get_model
from src.models.router import route_model
from src.prompts.prompts import (
    coach_prompt,
    evaluation_prompt,
    planner_prompt,
    question_prompt,
)
from src.tools.tool_executor import execute_tool_request

# --------------------------------------------------
# NODES
# --------------------------------------------------


def generate_question_node(llm: BaseChatModel):

    def generate_question(state: InterviewState):
        """Generate a question using the candidate's long-term memory."""

        # Retrieve persistent skill memory
        profiles = get_skill_profiles(state["topic"])

        memory_context = [
            {
                "skill": profile.skill,
                "attempts": profile.attempts,
                "average_score": round(profile.average_score, 2),
                "weaknesses": profile.weaknesses,
            }
            for profile in profiles
        ]

        # Request question bank
        request = ToolRequest(
            tool_name="get_question_bank",
            arguments={
                "topic": state["topic"],
            },
        )

        tool_response = execute_tool_request(request)

        if tool_response["status"] == "executed":
            mcp_result = tool_response["result"]

            question_bank = (
                mcp_result.structured_content.get("result", [])
                if mcp_result.structured_content
                else []
            )
        else:
            question_bank = []

        # Generate adaptive question
        structured_llm = llm.with_structured_output(InterviewQuestion).with_retry(
            stop_after_attempt=2
        )
        question_chain = question_prompt | structured_llm

        response = question_chain.invoke(
            {
                "topic": state["topic"],
                "previous_question": state["previous_question"],
                "question_bank": question_bank,
                "memory": memory_context,
                "interview_plan": (
                    state["interview_plan"].model_dump()
                    if state.get("interview_plan")
                    else "No plan yet. Start the interview normally."
                ),
            }
        )

        print("\nINTERVIEWER:")
        print(response.question)

        return {
            "question": response.question,
            "skill": response.skill,
        }

    return generate_question


def collect_answer(state: InterviewState):
    """Collect the candidate's answer."""

    answer = input("\nANSWER:\n")

    return {
        "answer": answer,
    }


def evaluate_answer_node(state: InterviewState):
    """Evaluate the candidate's answer."""

    decision = route_model(
        question=state["question"],
        answer=state["answer"],
    )

    llm = get_model(decision.model)

    structured_llm = llm.with_structured_output(Evaluation).with_retry(
        stop_after_attempt=2
    )
    evaluation_chain = evaluation_prompt | structured_llm

    try:
        evaluation = evaluation_chain.invoke(
            {
                "question": state["question"],
                "answer": state["answer"],
            }
        )
    except Exception as exc:
        if decision.model == ModelProvider.STRONG:
            raise RuntimeError("Strong evaluator failed.") from exc

        print(f"\nFast evaluator failed: {exc}")
        print("Falling back to strong model...")

        fallback_llm = get_model(ModelProvider.STRONG)
        fallback_structured_llm = fallback_llm.with_structured_output(
            Evaluation
        ).with_retry(stop_after_attempt=2)
        fallback_chain = evaluation_prompt | fallback_structured_llm

        evaluation = fallback_chain.invoke(
            {
                "question": state["question"],
                "answer": state["answer"],
            }
        )

    print("\nFEEDBACK:")
    print(f"Correctness: {evaluation.correctness}")
    print(f"Clarity: {evaluation.clarity}")
    print(f"Missing concepts: {evaluation.missing_concepts}")
    print(f"Improved answer: {evaluation.improved_answer}")
    print(f"Score: {VERDICT_TO_SCORE[evaluation.verdict]}/10")

    return {
        "evaluation": evaluation,
        "previous_question": state["question"],
    }


def ask_to_continue(state: InterviewState):
    """Ask whether the user wants another question."""

    command = input("\nPress Enter for the next question, or type 'quit': ")

    continue_interview = command.lower() not in {"quit", "exit"}

    return {
        "continue_interview": continue_interview,
    }


def plan_interview_node(state: InterviewState):
    """Decide the best next interview strategy."""

    profiles = get_skill_profiles(state["topic"])

    memory_context = [
        {
            "skill": profile.skill,
            "attempts": profile.attempts,
            "average_score": round(profile.average_score, 2),
            "weaknesses": profile.weaknesses,
        }
        for profile in profiles
    ]

    llm = get_model(ModelProvider.FAST)
    structured_llm = llm.with_structured_output(InterviewPlan).with_retry(
        stop_after_attempt=2
    )
    planner_chain = planner_prompt | structured_llm

    plan = planner_chain.invoke(
        {
            "topic": state["topic"],
            "skill": state["skill"],
            "question": state["question"],
            "evaluation": state["evaluation"].model_dump(),
            "memory": memory_context,
        }
    )

    print(f"\nPLAN: {plan.action}")
    print(f"Reason: {plan.reason}")

    return {"interview_plan": plan}


def coach_node(state: InterviewState):
    """Explain an identified weakness before the interview continues."""

    llm = get_model(ModelProvider.FAST)
    structured_llm = llm.with_structured_output(CoachOutput).with_retry(
        stop_after_attempt=2
    )
    chain = coach_prompt | structured_llm

    coaching = chain.invoke(
        {
            "question": state["question"],
            "answer": state["answer"],
            "evaluation": state["evaluation"].model_dump(),
        }
    )

    print("\nCOACH:")
    print(coaching.explanation)
    print(f"Review: {coaching.study_recommendation}")

    return {}


def human_review_node(state: InterviewState):
    """Allow a human to correct a low-confidence evaluation."""

    evaluation = state["evaluation"]

    human_verdict = interrupt(
        {
            "type": "evaluation_review",
            "message": "Evaluator confidence is low. Review the verdict.",
            "question": state["question"],
            "answer": state["answer"],
            "ai_verdict": evaluation.verdict,
        }
    )

    evaluation = evaluation.model_copy(update={"verdict": human_verdict})
    print(f"Reviewed score: {VERDICT_TO_SCORE[evaluation.verdict]}/10")

    return {
        "evaluation": evaluation,
    }


def update_memory_node(state: InterviewState):
    """Persist the accepted evaluation."""

    evaluation = state["evaluation"]

    update_skill_profile(
        topic=state["topic"],
        skill=state["skill"],
        verdict=evaluation.verdict,
        weakness=evaluation.missing_concepts,
    )

    return {}


# --------------------------------------------------
# ROUTING
# --------------------------------------------------


def route_after_continue(state: InterviewState):
    """Either generate another question or finish."""

    if state["continue_interview"]:
        return "continue"

    return "end"


def route_by_confidence(state: InterviewState):
    if state["evaluation"].confidence == "low":
        return "human_review"

    return "update_memory"


def route_after_plan(state: InterviewState):
    if state["interview_plan"].action == "coach":
        return "coach"

    return "continue"
