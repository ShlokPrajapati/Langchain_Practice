from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation"
)

model=ChatHuggingFace(llm=llm)
parser = JsonOutputParser()
# 1st prompt -> detailed report
# template = PromptTemplate(
#     template= ' Give me 5 facts about {topic} \n {format_instruction}',
#     input_variables=['topic'],
#     partial_variables={'format_instruction': parser.get_format_instructions()}
# )
template = PromptTemplate(
    template= ' Give me name, age and city of a football player \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
print(template.format())
# result = chain.invoke({'topic':'Rohit Sharma'})
result = chain.invoke({})
print(result)
