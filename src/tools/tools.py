import json
from pathlib import Path

from langchain_core.tools import tool

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

QUESTION_BANK_PATH = DATA_DIR / "question_bank.json"
TOPIC_NOTES_PATH = DATA_DIR / "topic_notes.json"


@tool
def get_question_bank(topic: str) -> list[str]:
    """Return interview questions for a given topic."""

    with QUESTION_BANK_PATH.open("r", encoding="utf-8") as file:
        question_bank = json.load(file)

    return {key.casefold(): value for key, value in question_bank.items()}.get(
        topic.strip().casefold(), []
    )


@tool
def get_topic_notes(topic: str) -> list[str]:
    """Return reference notes for a given interview topic."""

    if not TOPIC_NOTES_PATH.exists():
        return []

    with TOPIC_NOTES_PATH.open("r", encoding="utf-8") as file:
        topic_notes = json.load(file)

    return {key.casefold(): value for key, value in topic_notes.items()}.get(
        topic.strip().casefold(), []
    )


TOOLS = {
    get_question_bank.name: get_question_bank,
    get_topic_notes.name: get_topic_notes,
}
