import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, RunContextWrapper , function_tool , OpenAIChatCompletionsModel
from agents.run import RunConfig
import asyncio
from dataclasses import dataclass
from agents import set_default_openai_client, set_tracing_disabled


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")


external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)
set_default_openai_client(external_client)
set_tracing_disabled(True)

@dataclass  
class UserInfo:
    name : str
    uuid : int
    location : str = "Pakistan" 

@function_tool
async def dynamic_instructions(wrapper : RunContextWrapper[UserInfo]) -> str:
    return f"The user {wrapper.context.name} {wrapper.context.uuid} {wrapper.context.location} is 10 years old."
async def main():
    content = UserInfo(name="Abu Bakar" , uuid=12345678910)

    agent = Agent(
        name="Assistant",
        instructions = (dynamic_instructions, "This function returns the user information."),
        model=model,
    )

    result = await Runner.run(agent, "Hello.", context=content , run_config=RunConfig)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())