from dotenv import load_dotenv

load_dotenv()
from langfuse import get_client
from langfuse.langchain import CallbackHandler
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

    langfuse = get_client()
    langfuse_handler = CallbackHandler()

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

    config = {
        "configurable": {
            "thread_id": "interview-1",
        },
        "recursion_limit": 50,
        "callbacks": [langfuse_handler],
        "run_name": "adaptive-interview",
        "metadata": {
            "langfuse_session_id": "interview-1",
            "langfuse_tags": ["v8", "interview-agent"],
        },
    }
    result = graph.invoke(
        initial_state,
        config=config,
    )

    print(result)

    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value

        if interrupt_data.get("type") == "evaluation_review":
            print(f"\nAI verdict: {interrupt_data['ai_verdict']}")

            verdict = (
                input("Human verdict(excellent/good/partial/poor): ").strip().lower()
            )

            result = graph.invoke(
                Command(resume=verdict),
                config=config,
            )

        else:
            confirmation = input("\nApprove tool execution? (y/n): ").strip().lower()

            result = graph.invoke(
                Command(resume=confirmation == "y"),
                config=config,
            )
            print(result)

    langfuse.flush()


if __name__ == "__main__":
    main()
