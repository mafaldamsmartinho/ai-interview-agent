from graph import build_graph
from state import InterviewState


def main():
    print("\nAI Interview Practice Agent")
    print("---------------------------")

    graph = build_graph()

    topic = input(
        "\nChoose a topic "
        "(Machine Learning / Deep Learning / LLMs / "
        "AI Agents / Healthcare AI): "
    )

    initial_state: InterviewState = {
        "topic": topic,
        "question": "",
        "previous_question": "None",
        "answer": "",
        "feedback": "",
        "continue_interview": True,
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
