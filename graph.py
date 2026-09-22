from langgraph.graph import END, START, StateGraph

from nodes import (
    ask_to_continue,
    collect_answer,
    evaluate_answer,
    generate_question,
    route_after_answer,
    route_after_continue,
)
from state import InterviewState

builder = StateGraph(InterviewState)

builder.add_node("generate_question", generate_question)
builder.add_node("collect_answer", collect_answer)
builder.add_node("evaluate_answer", evaluate_answer)
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


graph = builder.compile()
