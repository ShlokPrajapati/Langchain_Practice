from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
import os

load_dotenv()

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation"
)

model=ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about Langchain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)