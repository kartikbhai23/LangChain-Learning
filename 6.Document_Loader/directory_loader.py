from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "6.Document_Loader/SubFolder",
    glob = "**/*.pdf",
    loader_cls = PyPDFLoader
)

documents = loader.load()

for document in documents:
    print("--------------------")
    print("Loaded Document:")
    print(type(document))
    print(document.page_content)
    print(document.metadata)
    print("--------------------")