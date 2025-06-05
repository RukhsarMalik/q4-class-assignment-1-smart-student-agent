import os
import asyncio
from agents import Agent , Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from dotenv import load_dotenv
from agents.run import RunConfig

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

external_client = AsyncOpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",  # Groq base URL
)
model = OpenAIChatCompletionsModel(
    model="llama3-70b-8192",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

agent = Agent(
    name = "Smart Student Agent",
    instructions= """
    You are a helpful academmic assistant designed to assist students with their studies.
    You can:
    1. Answer Academic questions clearly and conciesly.
    2. Provide study tips for students.
    3. Summarize short passages of text.
    only respond with useful and relvent content.
    """
    
)

# runner functions

async def run_agent_task(prompt: str):
    result = await Runner.run(agent, prompt, run_config=config)
    return result.final_output


async def answer_question():
    question = input("Enter your question: ")
    response = await run_agent_task(f"Answer this question:\n{question}")
    print("\nAnswer:", response)

async def provide_studey_tip():
    topic = input("Enter the topic for study tip: ")
    response = await run_agent_task(f"Provide study tip for {topic}.")
    print("\nStudy Tips:", response)

async def summarize_text():
    text = input("Enter the text to summarize: ")
    response = await run_agent_task(f"Summarize this text: \n{text}")
    print("\nSummary:", response)

# Main loop to interact with the agent

async def main():
    print("Welcome to the Smart Student Agent!")
    while True:
        print("\nOptions:")
        print("1. Answer a question")
        print("2. Provide study tips")
        print("3. Summarize text")
        print("4. Exit")
        
        choice = input("Choose an option (1-4): ")
        
        if choice == '1':
            await answer_question()
        elif choice == '2':
            await provide_studey_tip()
        elif choice == '3':
            await summarize_text()
        elif choice == '4':
            print("Exiting the Smart Student Agent. Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")
    
if __name__ == "__main__":
    asyncio.run(main())