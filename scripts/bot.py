import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ---- Config ----
VECTOR_STORE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "ghana_law_vectors")
EMBED_MODEL = "all-MiniLM-L6-v2"

# ---- Load FAISS ----
print("Loading vector store...")
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = FAISS.load_local(VECTOR_STORE_FOLDER, embedder, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# ---- Load LLM ----
llm = Ollama(model="gemma4", temperature=0.3)

# ---- Prompt ----
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a free legal information assistant specializing in Ghanaian law. You help ordinary Ghanaians understand their legal rights and options.

Use the context below to answer the question as helpfully as possible. Explain what the relevant law says, what rights or steps apply, and what the user should know.

If the context touches on the topic but does not fully answer the question, share what it does say and note what is unclear.
Only say you have no information if the context is completely unrelated to the question.

Rules:
- Plain language only — no legal jargon
- Keep answers concise and practical
- Always end with: "For serious matters, consult a Ghanaian lawyer or call the Legal Aid Commission on 0302 975 749 (lac.gov.gh)."

Context: {context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

docs_chain = create_stuff_documents_chain(llm, qa_prompt)

# ---- Simple manual session history ----
session_store: dict[str, list] = {}

def answer_query(query: str, session_id: str = "cli") -> tuple[str, list]:
    history = session_store.get(session_id, [])

    # Build retrieval query: combine last human turn + current query for context
    retrieval_query = query
    if history:
        last_human = next((m.content for m in reversed(history) if isinstance(m, HumanMessage)), "")
        if last_human and last_human.lower() != query.lower():
            retrieval_query = f"{last_human} {query}"

    docs = retriever.invoke(retrieval_query)

    answer = docs_chain.invoke({
        "input": query,
        "context": docs,
        "chat_history": history[-6:],  # last 3 exchanges
    })

    # Store turn in history
    if session_id not in session_store:
        session_store[session_id] = []
    session_store[session_id].extend([HumanMessage(content=query), AIMessage(content=answer)])
    session_store[session_id] = session_store[session_id][-10:]  # keep last 5 exchanges

    return answer, docs


def ask(query: str, session_id: str = "cli") -> None:
    print(f"\nQuestion: {query}")
    answer, docs = answer_query(query, session_id)
    print(f"\nAnswer:\n{answer}")
    print("\nSources:")
    for doc in docs:
        print(" -", doc.metadata.get("source", "Unknown"))


if __name__ == "__main__":
    while True:
        q = input("\nAsk me anything about Ghana Law (type 'exit' to quit): ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if q:
            ask(q)
