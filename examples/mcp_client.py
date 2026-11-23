"""A real MCP client: starts `gloss mcp` as a subprocess and talks to it over stdio,
the same way Claude Desktop or Claude Code would.

    python examples/e2e_shop.py ./demo                 # create a ledger to talk to
    python examples/mcp_client.py ./demo/gloss.db
"""

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DB = sys.argv[1] if len(sys.argv) > 1 else "gloss.db"

def show(result):
    data = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if data is None:
        data = [json.loads(c.text) for c in result.content]
    data = data.get("result", data) if isinstance(data, dict) and set(data) == {"result"} else data
    return json.dumps(data, ensure_ascii=False)

async def main():
    params = StdioServerParameters(command="gloss", args=["--db", f"sqlite:///{DB}", "mcp", "--agent", "claude"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print("connected to:", init.server_info.name if hasattr(init, "server_info") else init.serverInfo.name)
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])
            calls = [
                ("search_model", {"text": "what was revenue per country last month"}),
                ("search_model", {"text": "revnue"}),
                ("search_model", {"text": "email"}),
                ("search_model", {"text": "order", "kinds": ["metric"]}),
                ("get_model_context", {"datasets": ["shop.orders", "shop.customers"]}),
                ("review_status", {"subject": "field:shop.orders/amount"}),
            ]
            for name, args in calls:
                r = await session.call_tool(name, args)
                print(f"\n>>> {name}({json.dumps(args)})\n{show(r)}")

asyncio.run(main())
