import os
import traceback
import requests
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---- Config ----
VECTOR_STORE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "ghana_law_vectors")
EMBED_MODEL = "all-MiniLM-L6-v2"  # lightweight — fits in cloud free-tier RAM

# ---- Load FAISS ----
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = FAISS.load_local(VECTOR_STORE_FOLDER, embedder, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# ---- Load LLM: Groq in cloud, Ollama locally ----
groq_api_key = os.environ.get("GROQ_API_KEY")

if groq_api_key:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=groq_api_key,
    )
    print("Using Groq (cloud)")
else:
    from langchain_community.llms import Ollama
    llm = Ollama(model="gemma4", temperature=0.2)
    print("Using Ollama (local)")

# ---- Prompt ----
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a free legal information assistant specializing in Ghanaian law. You help ordinary Ghanaians understand their legal rights and options.

Use the context below to answer the question as helpfully as possible. Explain what the relevant law says, what rights or steps apply, and what the user should know.

If the context touches on the topic but does not fully answer the question, share what it does say and note what is unclear.
Only say you have no information if the context is completely unrelated to the question.

Rules:
- Plain language only — no legal jargon
- Keep answers concise and practical
- Always end with: "For serious matters, consult a Ghanaian lawyer or call the Legal Aid Commission free line: 0800-100-950."

Context: {context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def _format_docs(inputs: dict) -> dict:
    inputs["context"] = "\n\n".join(doc.page_content for doc in inputs["context"])
    return inputs

docs_chain = RunnablePassthrough.assign() | _format_docs | qa_prompt | llm | StrOutputParser()

# ---- Per-user session history (keyed by WhatsApp phone number) ----
session_store: dict[str, list] = {}

def answer_query(query: str, session_id: str) -> str:
    history = session_store.get(session_id, [])

    retrieval_query = query
    if history:
        last_human = next((m.content for m in reversed(history) if isinstance(m, HumanMessage)), "")
        if last_human and last_human.lower() != query.lower():
            retrieval_query = f"{last_human} {query}"

    docs = retriever.invoke(retrieval_query)

    answer = docs_chain.invoke({
        "input": query,
        "context": docs,
        "chat_history": history[-6:],
    })

    if session_id not in session_store:
        session_store[session_id] = []
    session_store[session_id].extend([HumanMessage(content=query), AIMessage(content=answer)])
    session_store[session_id] = session_store[session_id][-10:]

    return answer

# ---- Flask App ----
app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.form.get("Body", "").strip()
    sender = request.form.get("From", "unknown")
    resp = MessagingResponse()
    msg = resp.message()

    if not incoming_msg:
        msg.body("Please send a question about Ghanaian law and I'll do my best to help.")
        return str(resp)

    try:
        answer = answer_query(incoming_msg, sender)
        msg.body(answer)
    except Exception:
        print("ERROR:", traceback.format_exc())
        msg.body("Sorry, something went wrong. Please try again or contact the Legal Aid Commission of Ghana at 0800-100-950.")

    return str(resp)

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


# ---- Telegram Webhook ----
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

def send_telegram_message(chat_id: int, text: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # Telegram messages max 4096 chars — split if needed
    for chunk in [text[i:i+4096] for i in range(0, len(text), 4096)]:
        requests.post(url, json={"chat_id": chat_id, "text": chunk}, timeout=10)

@app.route("/telegram", methods=["POST"])
def telegram_reply():
    data = request.get_json(silent=True) or {}
    message = data.get("message") or data.get("edited_message")
    if not message:
        return jsonify({"ok": True})

    chat_id = message["chat"]["id"]
    text = message.get("text", "").strip()

    if not text:
        return jsonify({"ok": True})

    try:
        answer = answer_query(text, session_id=f"tg_{chat_id}")
        send_telegram_message(chat_id, answer)
    except Exception:
        print("TELEGRAM ERROR:", traceback.format_exc())
        send_telegram_message(chat_id, "Sorry, something went wrong. Please try again or call the Legal Aid Commission: 0800-100-950.")

    return jsonify({"ok": True})


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=5001, debug=False)
