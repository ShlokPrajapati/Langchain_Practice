from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
import os

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation"
)

model=ChatHuggingFace(llm=llm)
parser = StrOutputParser()
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template= 'Classify the sentiment of the following feedback text into positive or negative. Respond strictly in JSON format: {format_instruction} \n Feedback: {feedback}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)
branch_chain = RunnableBranch(
    ( lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    ( lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: 'could not found sentiment')
)
chain = classifier_chain | branch_chain 
raw_output = (prompt1 | model).invoke({'feedback': 'This is a worst phone'})
print("Raw Model Output:", raw_output)
result = chain.invoke({'feedback':'This is a beautiful phone'})

# print(result)

# chain.get_graph().print_ascii()
