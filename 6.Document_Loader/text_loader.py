from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

prompt = PromptTemplate(
    template = "Write a summary on the following poem.\n{poem}",
    input_variables = ["poem"]
)

parser = StrOutputParser()

loader = TextLoader("6.Document_Loader/cricket.txt", encoding="utf-8")

documents = loader.load()

print("--------------------")
print("Loaded Documents:")
print(type(documents))
print(len(documents))
print(documents[0].page_content)
print(documents[0].metadata)
print("--------------------")
chain = prompt | model | parser

print("Generating summary...")

response = chain.invoke({
    "poem": documents[0].page_content
})

print("Summary:")
print(response)