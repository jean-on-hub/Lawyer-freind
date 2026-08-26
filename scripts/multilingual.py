"""Voice input and Ghanaian-language support.

Two paths, deliberately separated by cost:

  English / Pidgin  ->  Groq Whisper        free, 2000 requests/day
  Twi / Ga / Ewe    ->  Khaya (Ghana NLP)   free tier is 100 calls/MONTH

Every Khaya call is metered against KHAYA_MONTHLY_QUOTA and refused once spent,
because a blown quota would otherwise surface to users as a stream of errors.
A Twi text conversation costs 2 calls (question in, answer out); a Twi voice
conversation costs 3. At the default quota that is ~50 text or ~33 voice
conversations per month.
"""

import os
import sqlite3
import requests

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "usage.db")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3-turbo")

KHAYA_API_KEY = os.environ.get("KHAYA_API_KEY")
KHAYA_BASE = os.environ.get("KHAYA_BASE", "https://translation-api.ghananlp.org")
KHAYA_MONTHLY_QUOTA = int(os.environ.get("KHAYA_MONTHLY_QUOTA", "100"))

# Khaya language codes. "en" stays on the free path and never costs a call.
LANGUAGES = {
    "english": "en",
    "twi": "tw",
    "ga": "gaa",
    "ewe": "ee",
    "dagbani": "dag",
    "fante": "fat",
    "frafra": "gur",
}
LANGUAGE_NAMES = {code: name for name, code in LANGUAGES.items()}


# ---- Storage: language preference + Khaya call metering ----

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS user_lang (session_id TEXT PRIMARY KEY, lang TEXT)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS khaya_calls (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            ts    TEXT DEFAULT (datetime('now')),
            kind  TEXT
        )
    """)
    conn.commit()
    return conn


def get_language(session_id: str) -> str:
    try:
        with _db() as conn:
            row = conn.execute("SELECT lang FROM user_lang WHERE session_id = ?", (session_id,)).fetchone()
        return row[0] if row else "en"
    except Exception:
        return "en"


def set_language(session_id: str, lang: str) -> None:
    try:
        with _db() as conn:
            conn.execute("INSERT OR REPLACE INTO user_lang (session_id, lang) VALUES (?,?)", (session_id, lang))
    except Exception:
        pass


def khaya_calls_this_month() -> int:
    try:
        with _db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM khaya_calls WHERE strftime('%Y-%m', ts) = strftime('%Y-%m', 'now')"
            ).fetchone()
        return row[0]
    except Exception:
        return 0


def khaya_budget_left(needed: int = 1) -> bool:
    """True if `needed` more Khaya calls fit inside this month's free quota."""
    return bool(KHAYA_API_KEY) and (khaya_calls_this_month() + needed) <= KHAYA_MONTHLY_QUOTA


def _record_khaya_call(kind: str) -> None:
    try:
        with _db() as conn:
            conn.execute("INSERT INTO khaya_calls (kind) VALUES (?)", (kind,))
    except Exception:
        pass


# ---- Free path: Groq Whisper (English / Ghanaian Pidgin) ----

def transcribe_english(audio: bytes, filename: str = "voice.ogg") -> str:
    """Transcribe with Whisper. Free, but English-only: Whisper does not support
    Twi, Ga, Ewe or Dagbani (excluded for >50% word error rate)."""
    r = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": (filename, audio)},
        data={"model": WHISPER_MODEL, "response_format": "json", "language": "en"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("text", "").strip()


# ---- Metered path: Khaya (Ghanaian languages) ----

def _khaya_headers(content_type: str = "application/json") -> dict:
    return {
        "Content-Type": content_type,
        "Cache-Control": "no-cache",
        "Ocp-Apim-Subscription-Key": KHAYA_API_KEY,
    }


def khaya_translate(text: str, pair: str) -> str:
    """Translate via Khaya. `pair` is e.g. "en-tw" or "tw-en". Costs one call."""
    r = requests.post(
        f"{KHAYA_BASE}/v1/translate",
        headers=_khaya_headers(),
        json={"in": text, "lang": pair},
        timeout=30,
    )
    r.raise_for_status()
    _record_khaya_call(f"translate:{pair}")
    out = r.json()
    return out if isinstance(out, str) else (out.get("translation") or out.get("out") or str(out))


def khaya_transcribe(audio: bytes, lang: str, content_type: str = "audio/mpeg") -> str:
    """Transcribe Ghanaian-language audio via Khaya. Costs one call."""
    r = requests.post(
        f"{KHAYA_BASE}/asr/v1/transcribe",
        headers=_khaya_headers(content_type),
        params={"language": lang},
        data=audio,
        timeout=90,
    )
    r.raise_for_status()
    _record_khaya_call(f"asr:{lang}")
    out = r.json()
    return out if isinstance(out, str) else (out.get("text") or str(out))


def to_english(text: str, lang: str) -> str:
    return text if lang == "en" else khaya_translate(text, f"{lang}-en")


def from_english(text: str, lang: str) -> str:
    return text if lang == "en" else khaya_translate(text, f"en-{lang}")
