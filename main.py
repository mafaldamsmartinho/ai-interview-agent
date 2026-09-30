from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.types import Command

from schemas import VERDICT_TO_SCORE
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

    topics = ["Machine Learning", "Deep Learning", "LLMs", "AI Agents", "Healthcare AI"]
    topic_names = {topic.casefold(): topic for topic in topics}
    while True:
        topic = input(f"\nChoose a topic ({' / '.join(topics)}): ").strip().casefold()
        if topic in topic_names:
            topic = topic_names[topic]
            break
        print("Please choose one of the listed topics.")

    initial_state: InterviewState = {
        "topic": topic,
        "skill": "",
        "question": "",
        "previous_question": "None",
        "answer": "",
        "evaluation": None,
        "continue_interview": True,
    }

    session_id = str(uuid4())
    config = {
        "configurable": {
            "thread_id": session_id,
        },
        "recursion_limit": 50,
        "callbacks": [langfuse_handler],
        "run_name": "adaptive-interview",
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_tags": ["interview-agent"],
        },
    }
    result = graph.invoke(
        initial_state,
        config=config,
    )

    while "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value

        if interrupt_data.get("type") == "evaluation_review":
            print(f"\nAI verdict: {interrupt_data['ai_verdict']}")

            while True:
                verdict = (
                    input("Human verdict (excellent/good/partial/poor): ")
                    .strip()
                    .lower()
                )
                if verdict in VERDICT_TO_SCORE:
                    break
                print("Please enter excellent, good, partial, or poor.")

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

    langfuse.flush()


if __name__ == "__main__":
    main()
