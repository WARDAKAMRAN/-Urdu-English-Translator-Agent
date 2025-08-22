# pip install openai-agents python-dotenv tensorflow rich

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool
from dotenv import load_dotenv
import os
import asyncio
from rich.console import Console
from rich.prompt import Prompt

# Load .env
load_dotenv()

# Setup console
console = Console()

# Fetch Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please define it in your .env file.")

# Setup Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Translation tool
@function_tool
def translate_text(text: str, target_language: str):
    return f"Translate '{text}' to {target_language}"

# Translator Agent
async def run_translation(text: str):
    agent = Agent(
        name='Translator Agent',
        instructions='You are a translator agent. Translate any Urdu text to English. Do not hallucinate.',
        tools=[translate_text]
    )
    response = await Runner.run(
        agent,
        input=text,
        run_config=config
    )
    return response

async def main():
    console.print("[bold cyan]🌐 Welcome to Urdu → English Translator[/bold cyan]\n")
    console.print("Type your Urdu text below. Type [bold red]'exit'[/bold red] to quit.\n")

    while True:
        user_input = Prompt.ask("[bold green]Meherbani kar ke Urdu me text likhein[/bold green]")
        if user_input.strip().lower() == "exit":
            console.print("\n[bold magenta]Shukriya! Program band ho gaya.[/bold magenta]")
            break
        if not user_input.strip():
            console.print("[bold yellow]Kripya kuch text likhein pehle.[/bold yellow]")
            continue

        translation = await run_translation(user_input)
        console.print("\n[bold blue]English Translation:[/bold blue]", style="yellow")
        console.print(f"{translation}\n")
        console.print("-" * 50 + "\n")

if __name__ == '__main__':
    asyncio.run(main())
