from langgraph.types import Command

from src.agent.graph import build_graph
from src.agent.state import InterviewState
from src.memory import models  # noqa: F401
from src.memory.database import Base, engine


def main():
    print("\nAI Interview Practice Agent")
    print("---------------------------")

    # Create persistent-memory tables if they do not exist
    Base.metadata.create_all(bind=engine)

    graph = build_graph()

    topic = input(
        "\nChoose a topic "
        "(Machine Learning / Deep Learning / LLMs / "
        "AI Agents / Healthcare AI): "
    )

    initial_state: InterviewState = {
        "topic": topic,
        "skill": "",
        "question": "",
        "previous_question": "None",
        "answer": "",
        "evaluation": None,
        "continue_interview": True,
    }

    config = {"configurable": {"thread_id": "interview-1"}}

    result = graph.invoke(
        initial_state,
        config=config,
    )

    print(result)

    while "__interrupt__" in result:
        confirmation = input(
            "\nSave this interview result? (y/n): "
        ).strip().lower()

        approved = confirmation == "y"

        result = graph.invoke(
            Command(resume=approved),
            config=config,
        )

        print(result)


if __name__ == "__main__":
    main()
