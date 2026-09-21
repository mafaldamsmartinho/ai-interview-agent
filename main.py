from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:1.7b",
    temperature=0.4,
)


question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a technical interviewer for AI and Machine Learning roles.

Ask exactly ONE interview question.

The question should:
- test understanding rather than memorization
- be appropriate for a junior-to-mid level AI/ML engineer
- be concise
- match the selected topic

Do not provide the answer.
Do not provide hints.
""",
        ),
        (
            "human",
            """
Topic: {topic}

Previous question:
{previous_question}

Ask a new question that is different from the previous question.
""",
        ),
    ]
)


question_chain = question_prompt | llm


evaluation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are evaluating an answer from a candidate in an AI/ML technical interview.

Evaluate the answer based on:

Correctness:
Is the technical explanation accurate?

Clarity:
Is the answer understandable and well explained?

Missing concepts:
Only identify missing concepts that are directly necessary to answer the interview question well.
Do not penalize the candidate for omitting advanced details, edge cases, related techniques, or possible follow-up topics unless the question explicitly asks for them.
If the answer already covers the expected scope, say:
Missing concepts: None significant.

Score:
What score should the answer have from 0-20. Evaluate the answer relative to the scope of the question, not relative to everything that could be said about the topic.

Keep the feedback concise.

Use this format:

Correctness:
<your evaluation>

Clarity:
<your evaluation>

Missing concepts:
<your evaluation>

Improved answer:
<a better version of the candidate's answer if necessary>

Score:
<your evaluation>
""",
        ),
        (
            "human",
            """
Interview question:
{question}

Candidate answer:
{answer}
""",
        ),
    ]
)


evaluation_chain = evaluation_prompt | llm


def generate_question(topic: str, previous_question: str) -> str:
    response = question_chain.invoke(
        {
            "topic": topic,
            "previous_question": previous_question,
        }
    )

    return str(response.content)


def evaluate_answer(question: str, answer: str) -> str:
    response = evaluation_chain.invoke(
        {
            "question": question,
            "answer": answer,
        }
    )

    return str(response.content)


def main():
    print("\nAI Interview Practice Agent")
    print("---------------------------")

    topic = input(
        "\nChoose a topic "
        "(Machine Learning / Deep Learning / LLMs / AI Agents / Healthcare AI): "
    )

    previous_question = "None"

    while True:
        question = generate_question(topic, previous_question)

        print("\nINTERVIEWER:")
        print(question)

        answer = input("\nANSWER:\n")

        if answer.lower() in {"quit", "exit"}:
            break

        feedback = evaluate_answer(question, answer)

        print("\nFEEDBACK:")
        print(feedback)

        previous_question = question

        command = input("\nPress Enter for the next question, or type 'quit' to stop: ")

        if command.lower() in {"quit", "exit"}:
            break


if __name__ == "__main__":
    main()
