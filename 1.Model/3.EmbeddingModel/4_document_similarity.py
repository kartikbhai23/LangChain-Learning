from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

#model to be used for generating embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

#documents to be embedded
documents = ["Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

#query to find the most similar document
query = "Who is the best Indian cricketer?"


#embed the documents and the query
document_vectors = embeddings.embed_documents(documents)
query_vector = embeddings.embed_query(query)


scores = cosine_similarity([query_vector], document_vectors)[0]

#find the index of the most similar document and its score
index, score = sorted(
    list(enumerate(scores)),
    key=lambda x: x[1]
)[-1]


#-------------------------------

# enumerate(scores) adds an index to each score.
# sorted() arranges the scores from smallest to largest.
# lambda x: x[1] tells Python to sort by the score.
# [-1] selects the last pair, which has the highest score.
# index stores the document's position, and score stores its similarity score

#-------------------------------

print(f"Query: {query}")
print(f"Most similar document: {documents[index]}")
print(f"Similarity score: {score}")