import os
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_FOLDER = os.path.join(os.path.dirname(__file__), "..", "Legal_documents")
VECTOR_STORE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "ghana_law_vectors")
EMBED_MODEL = "all-MiniLM-L6-v2"

docs = []

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF(s)")

for filename in tqdm(pdf_files, desc="Loading PDFs"):
    path = os.path.join(PDF_FOLDER, filename)
    try:
        loader = PyMuPDFLoader(path)
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = filename
        docs.extend(pages)
        print(f"  {filename} — {len(pages)} pages")
    except Exception as e:
        print(f"  ERROR loading {filename}: {e}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
print(f"\n{len(chunks)} total chunks from {len(docs)} pages")

print("Embedding and saving to FAISS...")
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vector_store = FAISS.from_documents(chunks, embedder)
vector_store.save_local(VECTOR_STORE_FOLDER)
print(f"Done. Vector store saved to '{VECTOR_STORE_FOLDER}'")
