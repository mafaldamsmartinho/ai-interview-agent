from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

PROMPTS_DIR = Path(__file__).parent


def load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


question_system_prompt = load_prompt("question_prompts/v1.4.0.txt")
evaluation_system_prompt = load_prompt("evaluation_prompts/v1.3.0.txt")
planner_system_prompt = load_prompt("planner_prompts/v1.0.0.txt")
coach_system_prompt = load_prompt("coach_prompts/v1.0.0.txt")


question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            question_system_prompt,
        ),
        (
            "human",
            """
Topic: {topic}

Previous question:
{previous_question}

Interview strategy:
{interview_plan}

Ask a new question that is different from the previous question.
""",
        ),
    ]
)


evaluation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            evaluation_system_prompt,
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

planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", planner_system_prompt),
        (
            "human",
            """
Topic: {topic}
Current skill: {skill}
Question: {question}

Evaluation:
{evaluation}

Candidate skill history:
{memory}
""",
        ),
    ]
)


coach_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", coach_system_prompt),
        (
            "human",
            """
Question:
{question}

Candidate answer:
{answer}

Evaluation:
{evaluation}
""",
        ),
    ]
)
