from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

class ModelOutput(BaseModel):
    key_used: list[str] = Field(..., description="List of keys used in the model output")
    summary: str = Field(..., description="Summary of the input text")
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="Sentiment of the input text")

structured_model = model.with_structured_output(ModelOutput)

result = structured_model.invoke("""Today, I am learning about TypedDict in Python. It is used to define the expected structure of a dictionary by specifying its keys and the types of their values.

I learned that TypedDict helps improve code readability, provides better type hints, and allows static type checkers to identify potential mistakes. However, it does not validate data at runtime by itself.

I am currently exploring these topics:

Simple TypedDict

Annotated TypedDict

Literal for restricting values to specific choices

Complex TypedDict structures, including required and optional keys

Advantages, disadvantages, and practical use cases

Current progress: I am building my understanding of TypedDict and how it can be used to represent structured data in Python, especially when working with  langgraph""")

print(result)