from prompts import evaluation_chain, question_chain
from state import InterviewState

# --------------------------------------------------
# NODES
# --------------------------------------------------


def generate_question(state: InterviewState):
    """Generate the next interview question."""

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


def collect_answer(state: InterviewState):
    """Collect the candidate's answer."""

    answer = input("\nANSWER:\n")

    return {
        "answer": answer,
    }


def evaluate_answer(state: InterviewState):
    """Evaluate the candidate's answer."""

    response = evaluation_chain.invoke(
        {
            "question": state["question"],
            "answer": state["answer"],
        }
    )

    feedback = str(response.content)

    print("\nFEEDBACK:")
    print(feedback)

    return {
        "feedback": feedback,
        "previous_question": state["question"],
    }


def ask_to_continue(state: InterviewState):
    """Ask whether the user wants another question."""

    command = input(
        "\nPress Enter for the next question, "
        "or type 'quit' to stop: "
    )

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
