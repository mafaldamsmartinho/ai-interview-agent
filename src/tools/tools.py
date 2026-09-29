import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

QUESTION_BANK_PATH = DATA_DIR / "question_bank.json"
HISTORY_PATH = DATA_DIR / "interview_history.json"
TOPIC_NOTES_PATH = DATA_DIR / "topic_notes.json"


@tool
def get_question_bank(topic: str) -> list[str]:
    """Return interview questions for a given topic."""

    with QUESTION_BANK_PATH.open("r", encoding="utf-8") as file:
        question_bank = json.load(file)

    return question_bank.get(topic, [])


@tool
def get_candidate_history() -> list[dict[str, Any]]:
    """Return the candidate's previous interview results."""

    if not HISTORY_PATH.exists():
        return []

    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@tool
def get_topic_notes(topic: str) -> list[str]:
    """Return reference notes for a given interview topic."""

    if not TOPIC_NOTES_PATH.exists():
        return []

    with TOPIC_NOTES_PATH.open("r", encoding="utf-8") as file:
        topic_notes = json.load(file)

    return topic_notes.get(topic, [])


@tool
def save_interview_result(
    topic: str,
    question: str,
    answer: str,
    score: int,
) -> str:
    """Save the result of an interview question."""

    if HISTORY_PATH.exists():
        with HISTORY_PATH.open("r", encoding="utf-8") as file:
            history = json.load(file)
    else:
        history = []

    history.append(
        {
            "topic": topic,
            "question": question,
            "answer": answer,
            "score": score,
        }
    )

    DATA_DIR.mkdir(exist_ok=True)

    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    return "Interview result saved."


TOOLS = {
    get_question_bank.name: get_question_bank,
    get_topic_notes.name: get_topic_notes,
    get_candidate_history.name: get_candidate_history,
    save_interview_result.name: save_interview_result,
}
