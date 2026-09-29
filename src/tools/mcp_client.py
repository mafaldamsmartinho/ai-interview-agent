import asyncio
import sys

from mcp import Client, StdioServerParameters

server = StdioServerParameters(
    command=sys.executable,
    args=["-m", "src.tools.mcp_server"],
)


async def call_mcp_tool_async(tool_name: str, arguments: dict):
    async with Client(server) as client:
        return await client.call_tool(tool_name, arguments)


def call_mcp_tool(tool_name: str, arguments: dict):
    """Synchronously call a tool exposed through MCP."""
    return asyncio.run(call_mcp_tool_async(tool_name, arguments))
