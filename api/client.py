import requests
import streamlit as st

def get_ollama_response(input_text):
    response=requests.post("http://localhost:8000/essay/invoke",
                           json={'input':{'topic':input_text}})
    
    return response.json()['output']

def get_ollama2_response(input_text):
    response=requests.post("http://localhost:8000/poem/invoke",
                           json={'input':{'topic':input_text}})
    json={'input':{'topic':input_text}}
    return response.json()['output']

st.title('Langchain Demo With MISTRAL API')
input_text=st.text_input("Write an essay on")

if input_text:
    st.write(get_ollama_response(input_text))
input_text1=st.text_input("Write a poem on")

if input_text1:
    st.write(get_ollama2_response(input_text1))