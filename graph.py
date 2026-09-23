from langgraph.graph import END, START, StateGraph

from models import ModelProvider, get_model
from nodes import (
    ask_to_continue,
    collect_answer,
    evaluate_answer_node,
    generate_question_node,
    route_after_answer,
    route_after_continue,
)
from state import InterviewState


def build_graph():
    builder = StateGraph(InterviewState)

    # Fix the fast llm for the question
    question_llm = get_model(ModelProvider.FAST)

    builder.add_node("generate_question", generate_question_node(question_llm))
    builder.add_node("collect_answer", collect_answer)
    builder.add_node("evaluate_answer", evaluate_answer_node)
    builder.add_node("ask_to_continue", ask_to_continue)

    # Entry point
    builder.add_edge(START, "generate_question")

    # Normal flow
    builder.add_edge("generate_question", "collect_answer")

    # Conditional route:
    # collect answer → evaluate OR finish
    builder.add_conditional_edges(
        "collect_answer",
        route_after_answer,
        {
            "evaluate": "evaluate_answer",
            "end": END,
        },
    )

    # Continue after evaluation
    builder.add_edge("evaluate_answer", "ask_to_continue")

    # Conditional route:
    # continue → another question
    # quit → END
    builder.add_conditional_edges(
        "ask_to_continue",
        route_after_continue,
        {
            "continue": "generate_question",
            "end": END,
        },
    )

    return builder.compile()
