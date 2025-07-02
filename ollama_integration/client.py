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

        tools_result  = await self.session.list_tools()

        print(f"Connected to MCP server and the following tools are available \n")

        for tool in tools_result.tools:
            print(f"{tool.name} : {tool.description}")


    def call_llama(self,prompt : str) -> str:
        # call the llama 3 model locally using ollama
        result = subprocess.run(
            ['ollama','run','llama3'],
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
        """Formats the prompt to instruct the LLM to select and call a tool."""
        tool_descriptions =  "\n".join( [ f"{tool['function']['name']} : {tool['function']['description']}" 
                                          for tool in tools])
       

        return f"""

        You are a helpful assistant with access to tools.str
        Available tools : {tool_descriptions} 


        Given the question : "{question}"

        Respond with a JSON object showing the tool or tools to call and the arguments like:
        {{'tool':'tool_name' , 'arguments':{{'arg1':'value1}}}}


        Answer only with JSON.
    
            """
    

    async def process_query(self,question:str) -> str:

        """Processes the user query through LLaMA Model -> MCP tool -> Result."""

        tools = await self.get_mcp_tools()
        prompt = self.format_prompt(tools,question)

        print("\n Sending prompt to LLaMA Model...\n")
        raw_output = self.call_llama(prompt)

        print("LLaMA Raw Output:")
        print(raw_output)

        try:
            tool_call = json.loads(raw_output)
            tool_name = tool_call['tool']
            arguments = tool_call['arguments']

        except Exception as e:
            return f" Error parsing LLaMA response: {e}"
        
        print ( f' \n Calling MCP tool {tool_name} with arguments {arguments}')

        result = await self.session.call_tool(tool_name, arguments)

        if not result.content:
            return " MCP tool did not return any content."
        
        context_text = result.content[0].text

        print("\nFeeding context back into LLaMA for final answer...\n")

        final_prompt = f"""You are a helpful assistant. Use the following context to answer the question.

        Context:
        {context_text}

        Question: {question}

        Answer:"""

        final_answer = self.call_llama(final_prompt)
        return final_answer



    async def cleanup(self):
        """Clean up any async context managers."""
        await self.exit_stack.aclose()




async def main():

    client = OllamaMCPClient()
    await client.connect_to_server("server.py")

    question = "What are THE POWERS OF THE COMMISSIONER?"

    response = await client.process_query(question)

    print( "Final response: \n ")
    print(response)

    await client.cleanup()



if __name__ == "__main__":
    asyncio.run(main())




    
