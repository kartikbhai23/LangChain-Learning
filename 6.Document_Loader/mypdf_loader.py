from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("6.Document_Loader/test_pdf.pdf")

documents = loader.load()

print("--------------------")
print("Loaded Documents:")
print(type(documents))
print(len(documents))
print(documents[0].page_content)
print(documents[0].metadata)
print("--------------------")