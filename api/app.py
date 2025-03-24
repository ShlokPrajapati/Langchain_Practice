from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
import uvicorn
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)
add_routes(
    app,
    Ollama(model="mistral"),
    path="/ollama"
)

llm = Ollama(model="mistral")

prompt1=ChatPromptTemplate.from_template("Write a concise 100-word essay explaining {topic}")
prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a creative poet, skilled in writing short and meaningful poems. Your poems are vivid, imaginative, and convey deep emotions in just a few lines."),
        ("user", "Write a short poem about {topic}.")
    ]
)
add_routes(
    app,
    prompt1|llm,
    path="/essay"
)
add_routes(
    app,
    prompt2|llm,
    path="/poem"
)
if __name__=="__main__":
    uvicorn.run(app, host="localhost", port=8000)