import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

load_dotenv()  

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
ENDPOINT_ID = os.getenv("VERTEX_AI_ENDPOINT_ID")
model = f"vertex_ai/openai/{ENDPOINT_ID}"
print (f"Model: {model}")

root_agent = LlmAgent(
    name="root_agent",
    model=LiteLlm(
        model=model,
    ),
    instruction=(
        """You are a helpful AI assistant designed to provide accurate and useful
        medical for patients needing help."""
    ),
    description="Answers questions about medical problems asked to you.",
)
