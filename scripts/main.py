import os
import pdfplumber
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

folder_path = os.path.join(os.path.dirname(__file__), "..", "Legal_documents")
PDF_FOLDER = os.path.abspath(folder_path)
VECTOR_STORE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "ghana_law_vectors")
EMBED_MODEL = "all-MiniLM-L6-v2"

# Load sentence embedding model
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

all_chunks = []

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def chunk_text(text, filename):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return [f"[{filename}] {chunk}" for chunk in chunks]

# List PDF files
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

print(f"📁 Found {len(pdf_files)} PDF(s) in '{PDF_FOLDER}'")

# Process with progress bar
for filename in tqdm(pdf_files, desc="📄 Processing PDFs"):
    try:
        pdf_path = os.path.join(PDF_FOLDER, filename)
        raw_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(raw_text, filename)
        all_chunks.extend(chunks)
        print(f"✅ {filename} — {len(chunks)} chunks")
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")

# Embed and save vectors
print("🔍 Embedding and saving to FAISS...")
vector_store = FAISS.from_texts(all_chunks, embedder)
vector_store.save_local(VECTOR_STORE_FOLDER)
print(f"✅ Done. Vector store saved to '{VECTOR_STORE_FOLDER}'")
