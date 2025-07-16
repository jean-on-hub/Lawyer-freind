from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Load FAISS vector store
VECTOR_STORE_FOLDER = "ghana_law_vectors"
EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

print("🔄 Loading vector store...")
db = FAISS.load_local(VECTOR_STORE_FOLDER, embedder, allow_dangerous_deserialization=True)

# Load LLaMA via Ollama
llm = Ollama(model="gemma3", temperature=0.3)

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    return_source_documents=True
)

def ask(query):
    print(f"\n💬 Question: {query}")
    response = qa_chain(query)
    
    print(f"\n🧠 Answer:\n{response['result']}")
    print("\n📚 Sources:")
    for doc in response["source_documents"]:
        print(" -", doc.metadata.get("source", "Unknown source"))

# Example interactive prompt
if __name__ == "__main__":
    while True:
        q = input("\nAsk me anything about Ghana Law (type 'exit' to quit): ")
        if q.lower() in ["exit", "quit"]:
            break
        ask(q)
