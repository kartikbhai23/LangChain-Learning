from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

jsocn_schema = {
    "type": "object",
    "properties": {
        "key_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of keys used in the model output"
        },  
        "summary": {
                "type": "string",
                "description": "Summary of the input text"
            },
        "sentiment": {
            "type": "string",
                "enum": ["positive", "negative", "neutral"],
                "description": "Sentiment of the input text"
            }
        },  
    "required": ["key_used", "summary", "sentiment"]
}

structured_model = model.with_structured_output(jsocn_schema)

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
