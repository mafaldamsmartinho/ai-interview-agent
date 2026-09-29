from langgraph.types import interrupt

from schemas import ToolRequest
from src.tools.mcp_client import call_mcp_tool
from src.tools.tool_guard import guard_tool_call
from src.tools.tools import TOOLS

MCP_TOOLS = {
    "get_question_bank": "question_bank",
    "get_topic_notes": "topic_notes",
}


def execute_tool_request(request: ToolRequest):

    decision = guard_tool_call(request.tool_name)

    if decision == "confirm":
        approved = interrupt(
            {
                "message": "Tool execution requires confirmation.",
                "tool": request.tool_name,
                "arguments": request.arguments,
            }
        )

        if not approved:
            return {
                "status": "rejected",
                "reason": "User rejected tool execution.",
            }

    # MCP tool
    if request.tool_name in MCP_TOOLS:
        result = call_mcp_tool(
            MCP_TOOLS[request.tool_name],
            request.arguments,
        )

    # Local tool
    elif request.tool_name in TOOLS:
        result = TOOLS[request.tool_name].invoke(request.arguments)

    else:
        return {
            "status": "denied",
            "reason": "Unknown tool.",
        }

    if decision == "allow_and_notify":
        print(f"Tool executed: {request.tool_name}")

    return {
        "status": "executed",
        "result": result,
    }
