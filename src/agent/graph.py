from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    ask_to_continue,
    coach_node,
    collect_answer,
    evaluate_answer_node,
    generate_question_node,
    human_review_node,
    plan_interview_node,
    route_after_continue,
    route_after_plan,
    route_by_confidence,
    save_result_node,
    update_memory_node,
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
    builder.add_node("human_review", human_review_node)
    builder.add_node("update_memory", update_memory_node)
    builder.add_node("plan_interview", plan_interview_node)
    builder.add_node("coach", coach_node)

    # Entry point
    builder.add_edge(START, "generate_question")

    # Normal flow
    builder.add_edge("generate_question", "collect_answer")

    # collect answer → evaluate
    builder.add_edge("collect_answer", "evaluate_answer")

    builder.add_conditional_edges(
        "evaluate_answer",
        route_by_confidence,
        {
            "human_review": "human_review",
            "update_memory": "update_memory",
        },
    )

    builder.add_edge("human_review", "update_memory")
    builder.add_edge("update_memory", "plan_interview")
    builder.add_conditional_edges(
        "plan_interview",
        route_after_plan,
        {
            "coach": "coach",
            "continue": "save_result",
        },
    )

    builder.add_edge("coach", "save_result")
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
