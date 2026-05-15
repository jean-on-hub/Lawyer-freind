from sentence_transformers import SentenceTransformer

print("Pre-downloading embedding model...")
SentenceTransformer("all-MiniLM-L6-v2")
print("Model cached successfully.")
