from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
import os

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm=HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz' : prompt2 | model | parser
})

merge_chain = prompt3 | model | parser
chain = parallel_chain | merge_chain
text = """
        Shivaji (born April 1627 or February 19, 1630, Shivner, Poona [now Pune], India—died April 3, 1680, Raigad) opposed the Mughal dynasty and founded the Maratha kingdom in 17th-century India. His kingdom’s security was based on religious toleration and on the functional integration of Brahmans, Marathas, and Prabhus.
        Shivaji was descended from a line of prominent nobles. At the time of his birth, about 1630, India was under Muslim rule: the Mughals in the north and the Muslim sultans of Bijapur and Golconda in the south. All three ruled by right of conquest, with no pretense that they had any obligations toward those who they ruled. Shivaji, whose ancestral estates were situated in the Deccan, in the realm of the Bijapur sultans, found the Muslim oppression and religious persecution of the Hindus so intolerable that, by the time he was 16, he convinced himself that he was the divinely appointed instrument of the cause of Hindu freedom—a conviction that was to sustain him throughout his life.

Collecting a band of followers, he began about 1655 to seize the weaker Bijapur outposts. In the process, he destroyed a few of his influential coreligionists, who had aligned themselves with the sultans. All the same, his daring and military skill, combined with his sternness toward the oppressors of the Hindus, won him much admiration. His depredations grew increasingly audacious, and he overcame the minor expeditions sent against him.
        """
result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()