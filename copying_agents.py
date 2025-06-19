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
config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True
)
async def main():

    agent = Agent(
        name="Assistant",
        instructions="Always greet the user using <function_call>greet_user</function_call>",
        model=model,
    )
    assistant = agent.clone(
        name = "Agent" , 
        instructions = "You are a helpfull agent",
        
    )
    result = await Runner.run(assistant, "Hello.", run_config=config)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())