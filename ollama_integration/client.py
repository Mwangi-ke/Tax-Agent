import asyncio
import json
from contextlib import AsyncExitStack
from typing import Optional,Any
import subprocess
from llama_cpp import Llama

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client



class OllamaMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()


    # connection to the server

    async def connect_to_server(self,server_script_path = 'server.py'):
        server_params = StdioServerParameters(command = 'python', args = [server_script_path])
        stdio = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream,write_stream = stdio
        self.session = await self.exit_stack.enter_async_context(ClientSession(read_stream,write_stream))
        await self.session.initialize()

        tools  = await self.session.list_tools()

        print(f"Connected to MCP server and the following tools are available \n")

        for tool in tools:
            print(f"{tool.name} : {tool.description}")


    def call_llama(self,prompt : str) -> str:
        # call the llama 3 model locally using ollama
        result = subprocess.run(
            ['ollamma','run','llama3'],
            input = prompt,
            text = True,
            capture_output= True
        )
        return result.stdout.strip()
            
    async def get_mcp_tools(self) -> list[dict[str,Any]]:
        """Get available tools from the MCP Server"""

        tools_result = await self.session.list_tools()

        return [{
            'type': 'function',
            "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]

    def format_prompt(self, tools: list[dict[str:Any]],question: str) -> str:

       
