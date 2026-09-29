from langchain_core.language_models.chat_models import BaseChatModel

from schemas import (
    VERDICT_TO_SCORE,
    Evaluation,
    InterviewQuestion,
    ToolRequest,
)
from src.agent.state import InterviewState
from src.memory.repository import get_skill_profiles, update_skill_profile
from src.models.models import get_model
from src.models.router import route_model
from src.prompts.prompts import evaluation_prompt, question_prompt
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
        structured_llm = llm.with_structured_output(InterviewQuestion)
        question_chain = question_prompt | structured_llm

        response = question_chain.invoke(
            {
                "topic": state["topic"],
                "previous_question": state["previous_question"],
                "question_bank": question_bank,
                "memory": memory_context,
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
    """Evaluate the candidate's answer and update persistent memory."""

    decision = route_model(
        question=state["question"],
        answer=state["answer"],
    )

    llm = get_model(decision.model)

    structured_llm = llm.with_structured_output(Evaluation)
    evaluation_chain = evaluation_prompt | structured_llm

    evaluation = evaluation_chain.invoke(
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

    # Update long-term skill memory
    update_skill_profile(
        topic=state["topic"],
        skill=state["skill"],
        verdict=evaluation.verdict,
        weakness=evaluation.missing_concepts,
    )

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


def save_result_node(state: InterviewState):
    evaluation = state["evaluation"]
    request = ToolRequest(
        tool_name="save_interview_result",
        arguments={
            "topic": state["topic"],
            "question": state["question"],
            "answer": state["answer"],
            "score": VERDICT_TO_SCORE[evaluation.verdict],
        },
    )

    result = execute_tool_request(request)

    return {"tool_result": result}


# --------------------------------------------------
# ROUTING
# --------------------------------------------------

def route_after_continue(state: InterviewState):
    """Either generate another question or finish."""

    if state["continue_interview"]:
        return "continue"

    return "end"


def route_by_score(state: InterviewState):
    score = VERDICT_TO_SCORE[state["evaluation"].verdict]

    if score >= 8:
        return "harder"

    if score >= 7:
        return "same"

    return "easier"
