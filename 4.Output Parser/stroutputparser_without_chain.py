from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

#1st prmpt --> detailed report
template1 = PromptTemplate(
    template = "write a detailed report on the following topic: {topic}",
    input_variables = ["topic"]
)

#2nd prompt --> summary 
template2 = PromptTemplate(
    template = "write a summary of the following text:\n{text}",
    input_variables = ["text"]
)

prompt1 = template1.invoke({"topic": "The impact of climate change on global agriculture."})
result1 = parser.invoke(model.invoke(prompt1))

prompt2 = template2.invoke({"text": result1})
result2 = parser.invoke(model.invoke(prompt2))

print("Detailed Report:\n", result1)
print("\nSummary:\n", result2)