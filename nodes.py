from langchain_core.language_models.chat_models import BaseChatModel

from prompts.prompts import evaluation_prompt, question_prompt
from schemas import Evaluation
from state import InterviewState

# --------------------------------------------------
# NODES
# --------------------------------------------------


def generate_question_node(llm: BaseChatModel):

    def generate_question(state: InterviewState):
        """Generate the next interview question."""

        question_chain = question_prompt | llm
        response = question_chain.invoke(
            {
                "topic": state["topic"],
                "previous_question": state["previous_question"],
            }
        )

        question = str(response.content)

        print("\nINTERVIEWER:")
        print(question)

        return {
            "question": question,
        }

    return generate_question


def collect_answer(state: InterviewState):
    """Collect the candidate's answer."""

    answer = input("\nANSWER:\n")

    return {
        "answer": answer,
    }


def evaluate_answer_node(llm: BaseChatModel):

    def evaluate_answer(state: InterviewState):
        """Evaluate the candidate's answer."""
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
        print(f"Score: {evaluation.score}/20")

        return {
            "feedback": evaluation,
            "previous_question": state["question"],
        }

    return evaluate_answer


def ask_to_continue(state: InterviewState):
    """Ask whether the user wants another question."""

    command = input("\nPress Enter for the next question, or type 'quit': ")

    continue_interview = command.lower() not in {"quit", "exit"}

    return {
        "continue_interview": continue_interview,
    }


# --------------------------------------------------
# ROUTING
# --------------------------------------------------


def route_after_answer(state: InterviewState):
    """Stop immediately if the candidate typed quit."""

    if state["answer"].lower() in {"quit", "exit"}:
        return "end"

    return "evaluate"


def route_after_continue(state: InterviewState):
    """Either generate another question or finish."""

    if state["continue_interview"]:
        return "continue"

    return "end"


def route_by_score(state: InterviewState):
    score = state["evaluation"].score

    if score >= 16:
        return "harder"

    if score >= 10:
        return "same"

    return "easier"
