from agents import Agent , Runner , AsyncOpenAI , RunContextWrapper ,  OpenAIChatCompletionsModel , function_tool
from agents.run import RunConfig
from dotenv import load_dotenv
import asyncio
import os
from dataclasses import dataclass
from pydantic import BaseModel
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key = GEMINI_API_KEY,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client=external_client
)
config = RunConfig(
    model = model,
    model_provider=external_client,
    tracing_disabled = True
)

class CalanderEvent(BaseModel):
    title : str
    date : str
    participants : list[str]

async def main():

    agent = Agent(
        name = "Assistant",
        instructions= "A Helpfull assisant",
        model = model,
        output_type=CalanderEvent

    )
    input1 = input("User : ")
    result = await Runner.run(agent , input1 , run_config=config ,)
    print("Agent : " , result.final_output)

if __name__ == "__main__":
    asyncio.run(main())    