import hashlib
import os
import re
import sqlite3
import time
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
    ("system", """You are a free legal assistant for Ghanaians. You give short, clear answers based on Ghanaian law.

IMPORTANT RULES:
1. If the question is broad or vague, ask ONE short clarifying question before answering. Do not dump a list of possibilities.
2. When you do answer, be direct and specific — no bullet-point encyclopaedias. 2–4 short paragraphs max.
3. Plain language only. No legal jargon.
4. Use the context below. If it only partly covers the topic, answer what you can and say what's unclear.
5. End every answer with: "For serious matters, call the Legal Aid Commission free: 0800-100-950."

Examples of good clarifying questions:
- "Are you buying or selling the land?"
- "Is this about a marriage, divorce, or inheritance?"
- "Who currently owns the land — an individual, a family, or a stool?"

Context: {context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def _format_docs(inputs: dict) -> dict:
    inputs["context"] = "\n\n".join(doc.page_content for doc in inputs["context"])
    return inputs

docs_chain = RunnablePassthrough.assign() | _format_docs | qa_prompt | llm | StrOutputParser()

# ---- Usage tracking (SQLite) ----
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "usage.db")

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ts        TEXT    DEFAULT (datetime('now')),
            channel   TEXT,
            user_hash TEXT,
            duration_ms INTEGER,
            error     INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn

def _log(channel: str, user_id: str, duration_ms: int, error: bool = False):
    user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO messages (channel, user_hash, duration_ms, error) VALUES (?,?,?,?)",
                (channel, user_hash, duration_ms, int(error))
            )
    except Exception:
        pass  # never let logging break the bot

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

    t0 = time.monotonic()
    try:
        answer = _clean_whatsapp(answer_query(incoming_msg, sender))
        msg.body(answer)
        _log("whatsapp", sender, int((time.monotonic() - t0) * 1000))
    except Exception:
        print("ERROR:", traceback.format_exc())
        _log("whatsapp", sender, int((time.monotonic() - t0) * 1000), error=True)
        msg.body("Sorry, something went wrong. Please try again or contact the Legal Aid Commission of Ghana at 0800-100-950.")

    return str(resp)

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

@app.route("/stats", methods=["GET"])
def stats():
    key = os.environ.get("STATS_KEY")
    if key and request.args.get("key") != key:
        return jsonify({"error": "unauthorized"}), 401
    with _db() as conn:
        totals = conn.execute("""
            SELECT channel,
                   COUNT(*)                             AS messages,
                   COUNT(DISTINCT user_hash)            AS unique_users,
                   SUM(error)                           AS errors,
                   ROUND(AVG(duration_ms))              AS avg_ms
            FROM messages GROUP BY channel
        """).fetchall()
        daily = conn.execute("""
            SELECT DATE(ts) AS date, channel, COUNT(*) AS messages
            FROM messages
            GROUP BY date, channel
            ORDER BY date DESC
            LIMIT 30
        """).fetchall()
    return jsonify({
        "totals": [dict(zip(["channel","messages","unique_users","errors","avg_ms"], r)) for r in totals],
        "last_30_days": [dict(zip(["date","channel","messages"], r)) for r in daily],
    })


# ---- Telegram Webhook ----
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

def _clean_whatsapp(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)                     # **bold** → *bold*
    text = re.sub(r'__(.+?)__', r'*\1*', text)                          # __bold__ → *bold*
    text = re.sub(r'\*([^*\n]+)\*', r'_\1_', text)                      # *italic* → _italic_
    text = re.sub(r'~~(.+?)~~', r'~\1~', text)                          # ~~strike~~ → ~strike~
    text = re.sub(r'`{3}[^\n]*\n?(.*?)`{3}', r'```\1```', text, flags=re.DOTALL)
    text = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)  # ## Header → *bold*
    text = re.sub(r'^\s*-\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _to_telegram_html(text: str) -> str:
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'```(?:\w+\n)?(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    text = re.sub(r'\*\*(.+?)\*\*|__(.+?)__', lambda m: f'<b>{m.group(1) or m.group(2)}</b>', text)
    text = re.sub(r'\*([^*\n]+)\*|_([^_\n]+)_', lambda m: f'<i>{m.group(1) or m.group(2)}</i>', text)
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'^#{1,6}\s+(.+)$', lambda m: f'<b>{m.group(1)}</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def send_telegram_message(chat_id: int, text: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    text = _to_telegram_html(text)
    for chunk in [text[i:i+4096] for i in range(0, len(text), 4096)]:
        requests.post(url, json={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"}, timeout=10)

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

    t0 = time.monotonic()
    try:
        answer = answer_query(text, session_id=f"tg_{chat_id}")
        send_telegram_message(chat_id, answer)
        _log("telegram", str(chat_id), int((time.monotonic() - t0) * 1000))
    except Exception:
        print("TELEGRAM ERROR:", traceback.format_exc())
        _log("telegram", str(chat_id), int((time.monotonic() - t0) * 1000), error=True)
        send_telegram_message(chat_id, "Sorry, something went wrong. Please try again or call the Legal Aid Commission: 0800-100-950.")

    return jsonify({"ok": True})


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=5001, debug=False)
