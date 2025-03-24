from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation"
)

model=ChatHuggingFace(llm=llm)

class Person(BaseModel):
    age: str = Field(description='Age of the person')
    city: str = Field(description='Name of the city the person belongs')


parser = PydanticOutputParser(pydantic_object=Person)
template = PromptTemplate(
    template= ' Generate the age and city of a {name} \n {format_instruction}',
    input_variables=['name'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
print(template)
result = chain.invoke({'name':'Lionel Messi'})

print(result)
