from mcp.server import MCPServer

from src.tools.tools import get_question_bank, get_topic_notes

mcp = MCPServer("interview-tools")


@mcp.tool()
def question_bank(topic: str) -> list[str]:
    """Return interview questions for a given topic."""
    return get_question_bank.invoke({"topic": topic})


@mcp.tool()
def topic_notes(topic: str) -> list[str]:
    """Return reference notes for a given interview topic."""
    return get_topic_notes.invoke({"topic": topic})


if __name__ == "__main__":
    mcp.run()
