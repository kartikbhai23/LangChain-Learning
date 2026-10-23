from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
loader = WebBaseLoader(url)

documents = loader.load()

chain = prompt | model | parser

# print("--------------------")
# print("Loaded Documents:")
# print(type(documents))
# print(len(documents))
# print(documents[0].page_content)
# print(documents[0].metadata)
# print("--------------------")

print(chain.invoke({
    "question": "What is the name  of the product?",
    "text": documents[0].page_content
}))