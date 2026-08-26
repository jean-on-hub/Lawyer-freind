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

import multilingual as ml

# ---- Config ----
VECTOR_STORE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "ghana_law_vectors")
EMBED_MODEL = "all-MiniLM-L6-v2"  # lightweight — fits in cloud free-tier RAM

# ---- Load FAISS ----
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = FAISS.load_local(VECTOR_STORE_FOLDER, embedder, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# ---- Load LLM: Groq in cloud, Ollama locally ----
groq_api_key = os.environ.get("GROQ_API_KEY")

# Model is env-overridable: Groq retires models periodically, and when that happens
# every request fails with a 404. Swapping GROQ_MODEL in .env is then a restart, not a deploy.
# Check available models: curl -H "Authorization: Bearer $GROQ_API_KEY" \
#   https://api.groq.com/openai/v1/models
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

if groq_api_key:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        api_key=groq_api_key,
    )
    print(f"Using Groq (cloud) — model: {GROQ_MODEL}")
else:
    from langchain_community.llms import Ollama
    llm = Ollama(model=os.environ.get("OLLAMA_MODEL", "gemma4"), temperature=0.2)
    print("Using Ollama (local)")

# ---- Prompt ----
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a free legal assistant for Ghanaians. You give short, clear answers based on Ghanaian law.

IMPORTANT RULES:
1. If the question is broad or vague, ask ONE short clarifying question before answering. Do not dump a list of possibilities.
2. When you do answer, be direct and specific — no bullet-point encyclopaedias. 2–4 short paragraphs max.
3. Plain language only. No legal jargon.
4. Use the context below. If it only partly covers the topic, answer what you can and say what's unclear.
5. End every answer with: "For serious matters, call the Legal Aid Commission on 0302 975 749 or visit lac.gov.gh."

Examples of good clarifying questions:
- "Are you buying or selling the land?"
- "Is this about a marriage, divorce, or inheritance?"
- "Who currently owns the land — an individual, a family, or a stool?"

ABOUT YOU — if the user asks what you can do, or asks for another language or for
voice, do not say you cannot. Tell them the exact word to send:
- Languages: they send just the word "twi", "ga", "ewe", "fante", "dagbani", "frafra",
  "kusaal", "yoruba" or "english"
- Voice replies: they send just the word "voice" (and "text" to go back)
- They can send you a voice note in English and you will understand it
- To start a new topic they send "new"

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
    # Added after the table shipped, so migrate rather than assume
    cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
    if "input_type" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN input_type TEXT DEFAULT 'text'")
    conn.commit()
    return conn

def _log(channel: str, user_id: str, duration_ms: int, error: bool = False, input_type: str = "text"):
    user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO messages (channel, user_hash, duration_ms, error, input_type) VALUES (?,?,?,?,?)",
                (channel, user_hash, duration_ms, int(error), input_type)
            )
    except Exception:
        pass  # never let logging break the bot

# ---- Per-user session history (keyed by WhatsApp phone number) ----
session_store: dict[str, list] = {}

RESET_TRIGGERS = {"new", "reset", "start over", "start fresh", "clear", "new topic",
                  "new conversation", "/new", "/reset"}
RESET_REPLY = ("Starting fresh! What legal question can I help you with?\n\n"
               "For serious matters, call the Legal Aid Commission on 0302 975 749 or visit lac.gov.gh.")

# Shown on /start, on "help", and once to every new user. The language names are
# listed as bare words because someone who reads little English can still
# recognise the name of their own language and send it back.
WELCOME = (
    "Akwaaba! I give free legal information about Ghanaian law — land, rent, work, "
    "marriage, inheritance, police and more.\n\n"
    "LANGUAGES — send one of these words:\n"
    "english · twi · ga · ewe · fante · dagbani · frafra · kusaal\n"
    "Or just write to me in your language and I will follow.\n\n"
    "VOICE NOTES — send one in English any time.\n"
    "For a Ghanaian language, send the language name first, then your voice note.\n"
    "Send \"voice\" if you want spoken replies, \"text\" to stop them.\n\n"
    "Send \"new\" to start a fresh topic.\n\n"
    "Ask me your question whenever you are ready.\n\n"
    "For serious matters, call the Legal Aid Commission on 0302 975 749 or visit lac.gov.gh."
)
START_TRIGGERS = {"/start", "start", "help", "/help", "menu", "/menu", "hi", "hello"}

def is_reset(text: str) -> bool:
    return text.lower().strip() in RESET_TRIGGERS

def is_start(text: str) -> bool:
    return text.lower().strip().rstrip("!.") in START_TRIGGERS

def is_first_contact(channel: str, user_id: str) -> bool:
    """True if this user has never sent a message before."""
    user_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]
    try:
        with _db() as conn:
            row = conn.execute(
                "SELECT 1 FROM messages WHERE channel = ? AND user_hash = ? LIMIT 1",
                (channel, user_hash),
            ).fetchone()
        return row is None
    except Exception:
        return False  # never greet twice by accident

def clear_session(session_id: str) -> None:
    session_store.pop(session_id, None)

# ---- Language selection ----
# Users can name a language explicitly; messages written in one are also detected
# automatically (see maybe_autodetect below).
LANGUAGE_HELP = ("Reply with a language to switch: " + ", ".join(ml.GHANA_LANGUAGES) + ".\n"
                 "Yoruba is also available. You can send a voice note in English.")

# Users write "can you speak twi?" far more often than a bare "twi", so short
# messages are matched on intent too. Legal questions that merely mention a
# language or the word voice are excluded by LEGAL_CONTEXT.
LEGAL_CONTEXT = re.compile(r"\b(law|legal|court|right|rights|act|land|rent|divorce|"
                           r"marriage|police|arrest|contract|will|inherit\w*)\b", re.I)

def _intent_word(text: str, options) -> str | None:
    """Match a bare command, or a short phrase asking for one of `options`."""
    cleaned = text.lower().strip().lstrip("/").rstrip("?!.")
    if cleaned in options:
        return cleaned
    words = re.findall(r"[a-z]+", cleaned)
    if len(words) > 6 or LEGAL_CONTEXT.search(cleaned):
        return None
    return next((w for w in words if w in options), None)

def handle_language_command(text: str, session_id: str) -> str | None:
    """Returns a confirmation if the message selects a language, else None."""
    if text.lower().strip().lstrip("/").rstrip("?") in ("language", "lang", "languages"):
        current = ml.LANGUAGE_NAMES.get(ml.get_language(session_id), "english")
        return f"You are using {current}.\n\n{LANGUAGE_HELP}"
    word = _intent_word(text, ml.LANGUAGES)
    if not word:
        return None

    code = ml.LANGUAGES[word]
    if code != "eng":
        if not ml.KHAYA_API_KEY:
            return ("Ghanaian languages aren't switched on yet — I can help in English for now. "
                    "You can send a voice note in English if that's easier.")
        if not ml.khaya_budget_left(2):
            return ("Sorry, this month's free quota for Ghanaian languages is used up. "
                    "I can still help in English — just ask your question.")
    ml.set_language(session_id, code)
    clear_session(session_id)
    if code == "eng":
        return "Switched to English. What legal question can I help you with?"
    confirm = f"Switched to {word.title()}. Ask your question."
    try:
        return ml.from_english(confirm, code)
    except Exception:
        return confirm

# ---- Automatic language detection ----
# Someone who only speaks Twi cannot read "reply with the word twi", so relying on
# the explicit command alone assumes the English literacy this feature exists to
# work around. Detection uses the LLM (free) rather than Khaya (100 calls/month),
# and is gated by a local check so English messages are never slowed down.
GHANAIAN_CHARS = set("ɛɔɖƒŋɣʒʋɲƐƆƉƑŊƔƷ")
ENGLISH_MARKERS = {"the", "is", "a", "to", "my", "i", "can", "what", "how", "of", "in",
                   "and", "me", "you", "for", "do", "if", "am", "was", "are", "he", "she",
                   "it", "they", "we", "have", "has", "want", "need", "please", "with"}

def looks_non_english(text: str) -> bool:
    if any(ch in GHANAIAN_CHARS for ch in text):
        return True
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 3:
        return False  # too short to judge without false positives
    return not any(w in ENGLISH_MARKERS for w in words)

def detect_language(text: str) -> str | None:
    """Ask the LLM which language this is. Returns a LANGUAGES key, or None."""
    options = ", ".join(ml.GHANA_LANGUAGES + ["yoruba"])
    try:
        result = llm.invoke(
            f"Which language is this message written in? Answer with ONE word from "
            f"this list and nothing else: {options}. If you are unsure, answer english.\n\n"
            f"Message: {text}"
        )
        word = getattr(result, "content", result).strip().lower().strip(".\"'")
        return word if word in ml.LANGUAGES else None
    except Exception:
        print("DETECT ERROR:", traceback.format_exc())
        return None

def maybe_autodetect(text: str, session_id: str) -> str | None:
    """Switch language in any direction when the user changes language mid-chat.

    Someone set to Twi who then writes English — or Ewe — should be followed, not
    left stranded. Detection is free (LLM), so the only cost is latency, which is
    why plainly-English messages from English users skip it entirely.
    """
    current = ml.get_language(session_id)
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 3 and not any(ch in GHANAIAN_CHARS for ch in text):
        return None  # too short to judge; stay where we are
    if current == "eng" and not looks_non_english(text):
        return None  # already English and still English: nothing to do

    detected = detect_language(text)
    if not detected:
        return None
    code = ml.LANGUAGES[detected]
    if code == current:
        return None
    if code != "eng" and not ml.khaya_budget_left(2):
        return None  # cannot serve that language right now
    ml.set_language(session_id, code)
    return detected

CONTACT_MARKER = "For serious matters"

def _split_contact(answer: str) -> tuple[str, str]:
    """Separate the Legal Aid sign-off so it can bypass translation intact."""
    idx = answer.find(CONTACT_MARKER)
    if idx == -1:
        return answer, ""
    return answer[:idx].rstrip(), answer[idx:].strip()

def answer_in_language(query: str, session_id: str) -> tuple[str, str]:
    """Run the English RAG pipeline, translating in and out when needed.

    Returns (reply_to_send, english_answer). The English is kept because machine
    translation of legal text can shift meaning, so translated replies carry the
    original alongside — and because TTS is English-only.

    Falls back to English rather than failing if Khaya errors or the quota is
    spent: a reply in the wrong language beats no reply at all.
    """
    lang = ml.get_language(session_id)
    if lang == "eng":
        answer = answer_query(query, session_id)
        return answer, answer

    if not ml.khaya_budget_left(2):
        answer = answer_query(query, session_id)
        return answer + "\n\n(This month's Ghanaian-language quota is used up, so this reply is in English.)", answer

    try:
        english_q = ml.to_english(query, lang)
        answer = answer_query(english_q, session_id)
        # Never translate the contact details: a round trip turned lac.gov.gh into
        # lc.gov.gh, which would send people to a domain that does not exist.
        body, contact = _split_contact(answer)
        translated = ml.from_english(body, lang)
        reply = f"{translated}\n\n{contact}" if contact else translated
        # English shown alongside so a bilingual reader can catch a bad translation
        return f"{reply}\n\n———\n(English)\n{answer}", answer
    except Exception:
        print("TRANSLATION ERROR:", traceback.format_exc())
        answer = answer_query(query, session_id)
        return answer, answer

# ---- Voice replies ----
# Orpheus caps input at 200 characters, so we speak a short summary rather than
# burning several calls narrating a full legal answer.
VOICE_PROMPT = ("\n\n---\nWant replies as voice notes? Reply *voice*. "
                "To keep replies as text, reply *text*.")

def spoken_summary(answer: str) -> str:
    if len(answer) <= ml.TTS_MAX_CHARS:
        return answer
    try:
        result = llm.invoke(
            "Summarise this legal answer in ONE spoken sentence under 180 characters. "
            "Plain words, no lists, no phone numbers:\n\n" + answer
        )
        summary = getattr(result, "content", result).strip()
        if summary:
            return summary[:ml.TTS_MAX_CHARS]
    except Exception:
        print("SUMMARY ERROR:", traceback.format_exc())
    cut = answer[:ml.TTS_MAX_CHARS]
    return cut.rsplit(".", 1)[0] + "." if "." in cut else cut

def wants_voice_reply(session_id: str) -> bool:
    return ml.get_voice_pref(session_id) == "voice" and ml.tts_budget_left()

def handle_voice_command(text: str, session_id: str, supports_voice: bool = True) -> str | None:
    word = _intent_word(text, ("voice", "audio", "text"))
    if not word:
        return None
    if word == "audio":
        word = "voice"
    if word == "text":
        ml.set_voice_pref(session_id, "text")
        return "Okay — text replies only."
    if not supports_voice:
        return "Voice replies aren't available on WhatsApp, so I'll keep replying with text."
    if not ml.tts_budget_left():
        return "Voice replies aren't available right now. I'll keep replying with text."
    ml.set_voice_pref(session_id, "voice")
    return "Okay — I'll send a short voice note with each answer, plus the full text."

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

# ---- Voice ----
# Whisper only knows English, so a Twi or Ewe voice note transcribes to nothing.
# The user cannot know that, so say what actually fixes it.
VOICE_FAILED = ("I couldn't understand that voice note.\n\n"
                "If you spoke Twi, Ga, Ewe, Fante, Dagbani, Frafra or Kusaal, "
                "first send me the name of the language (for example: ewe), "
                "then send your voice note again.\n\n"
                "You can also type your question instead.")

def transcribe_voice(audio: bytes, session_id: str, filename: str) -> str:
    """English voice is free (Whisper); Ghanaian languages are metered (Khaya)."""
    lang = ml.get_language(session_id)
    if lang == "eng" or not ml.khaya_budget_left(3):
        # Whisper is English-only. For a non-English user with no quota left it is
        # still the better attempt than nothing, since many users code-switch.
        return ml.transcribe_english(audio, filename)
    # Label the audio honestly — both Telegram and Twilio send OGG/Opus
    content_type = "audio/wav" if filename.endswith(".wav") else "audio/ogg"
    return ml.khaya_transcribe(audio, lang, content_type=content_type)

# ---- Flask App ----
app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.form.get("Body", "").strip()
    sender = request.form.get("From", "unknown")
    resp = MessagingResponse()
    msg = resp.message()

    # Voice note: Twilio sends media as a URL that needs account credentials
    if request.form.get("NumMedia", "0") != "0" and not incoming_msg:
        media_url = request.form.get("MediaUrl0", "")
        content_type = request.form.get("MediaContentType0", "")
        if media_url and content_type.startswith("audio"):
            try:
                audio = requests.get(
                    media_url,
                    auth=(os.environ.get("TWILIO_ACCOUNT_SID", ""), os.environ.get("TWILIO_AUTH_TOKEN", "")),
                    timeout=30,
                ).content
                incoming_msg = transcribe_voice(audio, sender, "voice.ogg")
            except Exception:
                print("VOICE ERROR:", traceback.format_exc())
                _log("whatsapp", sender, 0, error=True, input_type="voice")
                msg.body(VOICE_FAILED)
                return str(resp)
        if not incoming_msg:
            _log("whatsapp", sender, 0, error=True, input_type="voice")
            msg.body(VOICE_FAILED)
            return str(resp)

    if not incoming_msg:
        msg.body("Please send a question about Ghanaian law and I'll do my best to help.")
        return str(resp)

    if is_start(incoming_msg):
        clear_session(sender)
        msg.body(WELCOME)
        return str(resp)

    if is_reset(incoming_msg):
        clear_session(sender)
        msg.body(RESET_REPLY)
        return str(resp)

    if is_first_contact("whatsapp", sender):
        msg.body(WELCOME)
        msg = resp.message()  # answer follows as a second message

    lang_reply = handle_language_command(incoming_msg, sender)
    if lang_reply:
        msg.body(lang_reply)
        return str(resp)

    voice_reply = handle_voice_command(incoming_msg, sender, supports_voice=False)
    if voice_reply:
        msg.body(voice_reply)
        return str(resp)

    t0 = time.monotonic()
    try:
        # Text only on WhatsApp: Twilio needs a publicly hosted URL for media,
        # which the bot has no way to serve.
        detected = maybe_autodetect(incoming_msg, sender)
        answer, _ = answer_in_language(incoming_msg, sender)
        if detected:
            answer += f'\n\n(Detected {detected.title()}. Reply "english" for English.)'
        msg.body(_clean_whatsapp(answer))
        _log("whatsapp", sender, int((time.monotonic() - t0) * 1000))
    except Exception:
        print("ERROR:", traceback.format_exc())
        _log("whatsapp", sender, int((time.monotonic() - t0) * 1000), error=True)
        msg.body("Sorry, something went wrong. Please try again, or contact the Legal Aid Commission on 0302 975 749 (lac.gov.gh).")

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
                   COUNT(*)                                            AS messages,
                   COUNT(DISTINCT user_hash)                           AS unique_users,
                   SUM(error)                                          AS errors,
                   ROUND(AVG(duration_ms))                             AS avg_ms,
                   SUM(CASE WHEN input_type = 'voice' THEN 1 ELSE 0 END) AS voice_messages,
                   ROUND(AVG(CASE WHEN input_type = 'voice' THEN duration_ms END)) AS voice_avg_ms
            FROM messages GROUP BY channel
        """).fetchall()
        daily = conn.execute("""
            SELECT DATE(ts) AS date, channel, COUNT(*) AS messages
            FROM messages
            GROUP BY date, channel
            ORDER BY date DESC
            LIMIT 30
        """).fetchall()
    used = ml.khaya_calls_this_month()
    tts_used = ml.tts_calls_this_month()
    return jsonify({
        "totals": [dict(zip(["channel","messages","unique_users","errors","avg_ms",
                             "voice_messages","voice_avg_ms"], r)) for r in totals],
        "last_30_days": [dict(zip(["date","channel","messages"], r)) for r in daily],
        "khaya_quota": {
            "used_this_month": used,
            "monthly_quota": ml.KHAYA_MONTHLY_QUOTA,
            "remaining": max(0, ml.KHAYA_MONTHLY_QUOTA - used),
            "enabled": bool(ml.KHAYA_API_KEY),
        },
        "tts_quota": {
            "used_this_month": tts_used,
            "monthly_quota": ml.TTS_MONTHLY_QUOTA,
            "remaining": max(0, ml.TTS_MONTHLY_QUOTA - tts_used),
            "enabled": ml.TTS_ENABLED,
        },
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

def download_telegram_file(file_id: str) -> bytes:
    meta = requests.get(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile",
        params={"file_id": file_id}, timeout=15,
    ).json()
    path = meta["result"]["file_path"]
    r = requests.get(f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{path}", timeout=60)
    r.raise_for_status()
    return r.content

def send_telegram_voice(chat_id: int, wav: bytes) -> None:
    """Orpheus returns WAV, which is not Telegram's voice-note format (OGG/Opus),
    so this goes out via sendAudio and appears as a playable audio clip."""
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendAudio",
        data={"chat_id": chat_id, "title": "Answer"},
        files={"audio": ("answer.wav", wav, "audio/wav")},
        timeout=60,
    )

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
    session_id = f"tg_{chat_id}"
    text = message.get("text", "").strip()

    # Timed from here so voice includes download + transcription, otherwise a
    # voice note is indistinguishable from a text one in /stats
    t0 = time.monotonic()

    # Voice note or forwarded audio: Telegram gives a file_id to resolve first
    voice = message.get("voice") or message.get("audio")
    sent_voice = bool(voice)
    if not text and voice:
        try:
            audio = download_telegram_file(voice["file_id"])
            text = transcribe_voice(audio, session_id, "voice.ogg")
        except Exception:
            print("VOICE ERROR:", traceback.format_exc())
            _log("telegram", str(chat_id), int((time.monotonic() - t0) * 1000),
                 error=True, input_type="voice")
            send_telegram_message(chat_id, VOICE_FAILED)
            return jsonify({"ok": True})
        if not text:
            # Empty transcription: almost always a non-English voice note
            _log("telegram", str(chat_id), int((time.monotonic() - t0) * 1000),
                 error=True, input_type="voice")
            send_telegram_message(chat_id, VOICE_FAILED)
            return jsonify({"ok": True})

    if not text:
        return jsonify({"ok": True})

    if is_start(text):
        clear_session(session_id)
        send_telegram_message(chat_id, WELCOME)
        return jsonify({"ok": True})

    if is_reset(text):
        clear_session(session_id)
        send_telegram_message(chat_id, RESET_REPLY)
        return jsonify({"ok": True})

    # Greet a first-time user before answering, so they learn the options exist
    if is_first_contact("telegram", str(chat_id)):
        send_telegram_message(chat_id, WELCOME)

    lang_reply = handle_language_command(text, session_id)
    if lang_reply:
        send_telegram_message(chat_id, lang_reply)
        return jsonify({"ok": True})

    voice_reply = handle_voice_command(text, session_id)
    if voice_reply:
        send_telegram_message(chat_id, voice_reply)
        return jsonify({"ok": True})

    try:
        detected = maybe_autodetect(text, session_id)
        answer, english = answer_in_language(text, session_id)
        if detected:
            answer += f'\n\n(Detected {detected.title()}. Reply "english" for English.)'

        if wants_voice_reply(session_id):
            try:
                send_telegram_voice(chat_id, ml.synthesize_english(spoken_summary(english)))
            except Exception:
                print("TTS ERROR:", traceback.format_exc())

        # Ask once, only after they have actually used voice, so it stays relevant
        if sent_voice and not ml.was_asked_about_voice(session_id):
            ml.mark_asked_about_voice(session_id)
            answer += VOICE_PROMPT

        send_telegram_message(chat_id, answer)
        _log("telegram", str(chat_id), int((time.monotonic() - t0) * 1000),
             input_type="voice" if sent_voice else "text")
    except Exception:
        print("TELEGRAM ERROR:", traceback.format_exc())
        _log("telegram", str(chat_id), int((time.monotonic() - t0) * 1000), error=True,
             input_type="voice" if sent_voice else "text")
        send_telegram_message(chat_id, "Sorry, something went wrong. Please try again, or call the Legal Aid Commission on 0302 975 749 (lac.gov.gh).")

    return jsonify({"ok": True})


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=5001, debug=False)
