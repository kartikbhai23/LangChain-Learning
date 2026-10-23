from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="6.Document_Loader/data.csv", encoding="utf-8")

documents = loader.load()

print("--------------------")
print("Loaded Documents:")
print(type(documents))
print(len(documents))
print(documents[0].page_content)
print(documents[0].metadata)
print("--------------------")