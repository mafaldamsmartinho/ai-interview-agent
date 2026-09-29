from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    ask_to_continue,
    collect_answer,
    evaluate_answer_node,
    generate_question_node,
    route_after_continue,
    save_result_node,
)
from src.agent.state import InterviewState
from src.models.models import ModelProvider, get_model


def build_graph():
    builder = StateGraph(InterviewState)

    # Fix the fast llm for the question
    question_llm = get_model(ModelProvider.FAST)

    builder.add_node("generate_question", generate_question_node(question_llm))
    builder.add_node("collect_answer", collect_answer)
    builder.add_node("evaluate_answer", evaluate_answer_node)
    builder.add_node("save_result", save_result_node)
    builder.add_node("ask_to_continue", ask_to_continue)

    # Entry point
    builder.add_edge(START, "generate_question")

    # Normal flow
    builder.add_edge("generate_question", "collect_answer")

    # collect answer → evaluate
    builder.add_edge("collect_answer", "evaluate_answer")

    # Save after evaluation
    builder.add_edge("evaluate_answer", "save_result")

    # Continue after saving
    builder.add_edge("save_result", "ask_to_continue")

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

    checkpointer = InMemorySaver()

    return builder.compile(checkpointer=checkpointer)
