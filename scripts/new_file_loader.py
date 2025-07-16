from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os

docs = []
pdf_folder = "Legal_documents"  # Your folder with PDFs

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_folder, filename)
        loader = PyPDFLoader(path)
        pages = loader.load()

        # Set the source in metadata
        for page in pages:
            page.metadata["source"] = filename

        docs.extend(pages)

# Now chunk the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(docs)
