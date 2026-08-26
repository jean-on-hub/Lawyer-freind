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

# Orpheus TTS. English only, ~100 free requests, and it rejects input over 200
# characters — so we speak a short summary and send the full answer as text.
TTS_MODEL = os.environ.get("TTS_MODEL", "canopylabs/orpheus-v1-english")
TTS_VOICE = os.environ.get("TTS_VOICE", "Hannah")
TTS_MONTHLY_QUOTA = int(os.environ.get("TTS_MONTHLY_QUOTA", "100"))
TTS_MAX_CHARS = 200
TTS_ENABLED = os.environ.get("TTS_ENABLED", "true").lower() not in ("false", "0", "no")

# Official Khaya codes, from GET /v1/languages — do not guess these: the API
# accepts "tw" as an alias for Twi but rejects invented codes, so a wrong guess
# fails only for that one language, silently, in production.
# "eng" stays on the free path and never costs a call.
LANGUAGES = {
    "english": "eng",
    "twi": "twi",
    "ga": "gaa",
    "ewe": "ewe",
    "fante": "fat",
    "dagbani": "dag",
    "frafra": "gur",     # Khaya calls it Gurune; Frafra is the common Ghanaian name
    "gurune": "gur",
    "kusaal": "kus",
    "yoruba": "yor",
    # Also offered by Khaya, kept for anyone who needs them: East African
    "kikuyu": "kik",
    "luo": "luo",
    "kimeru": "mer",
}
# Ghanaian languages first — these are what the help text advertises
GHANA_LANGUAGES = ["english", "twi", "ga", "ewe", "fante", "dagbani", "frafra", "kusaal"]
LANGUAGE_NAMES = {
    "eng": "english", "twi": "twi", "gaa": "ga", "ewe": "ewe", "fat": "fante",
    "dag": "dagbani", "gur": "frafra", "kus": "kusaal", "yor": "yoruba",
    "kik": "kikuyu", "luo": "luo", "mer": "kimeru",
}


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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_voice (
            session_id TEXT PRIMARY KEY,
            pref       TEXT,
            asked      INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE TABLE IF NOT EXISTS tts_calls (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT (datetime('now')))")
    conn.commit()
    return conn


def get_language(session_id: str) -> str:
    try:
        with _db() as conn:
            row = conn.execute("SELECT lang FROM user_lang WHERE session_id = ?", (session_id,)).fetchone()
        return row[0] if row else "eng"
    except Exception:
        return "eng"


def set_language(session_id: str, lang: str) -> None:
    try:
        with _db() as conn:
            conn.execute("INSERT OR REPLACE INTO user_lang (session_id, lang) VALUES (?,?)", (session_id, lang))
    except Exception:
        pass


def get_voice_pref(session_id: str) -> str | None:
    """'voice', 'text', or None when the user has never chosen."""
    try:
        with _db() as conn:
            row = conn.execute("SELECT pref FROM user_voice WHERE session_id = ?", (session_id,)).fetchone()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def set_voice_pref(session_id: str, pref: str) -> None:
    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO user_voice (session_id, pref, asked) VALUES (?,?,1) "
                "ON CONFLICT(session_id) DO UPDATE SET pref = excluded.pref, asked = 1",
                (session_id, pref),
            )
    except Exception:
        pass


def was_asked_about_voice(session_id: str) -> bool:
    try:
        with _db() as conn:
            row = conn.execute("SELECT asked FROM user_voice WHERE session_id = ?", (session_id,)).fetchone()
        return bool(row and row[0])
    except Exception:
        return True  # on error, stay quiet rather than nag


def mark_asked_about_voice(session_id: str) -> None:
    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO user_voice (session_id, asked) VALUES (?,1) "
                "ON CONFLICT(session_id) DO UPDATE SET asked = 1",
                (session_id,),
            )
    except Exception:
        pass


def tts_calls_this_month() -> int:
    try:
        with _db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM tts_calls WHERE strftime('%Y-%m', ts) = strftime('%Y-%m', 'now')"
            ).fetchone()
        return row[0]
    except Exception:
        return 0


def tts_budget_left() -> bool:
    return TTS_ENABLED and bool(GROQ_API_KEY) and tts_calls_this_month() < TTS_MONTHLY_QUOTA


def synthesize_english(text: str) -> bytes:
    """Speak up to TTS_MAX_CHARS of English. Returns WAV bytes. Costs one call."""
    r = requests.post(
        "https://api.groq.com/openai/v1/audio/speech",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": TTS_MODEL,
            "input": text[:TTS_MAX_CHARS],
            "voice": TTS_VOICE,
            "response_format": "wav",
        },
        timeout=60,
    )
    r.raise_for_status()
    try:
        with _db() as conn:
            conn.execute("INSERT INTO tts_calls DEFAULT VALUES")
    except Exception:
        pass
    return r.content


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


def khaya_transcribe(audio: bytes, lang: str, content_type: str = "audio/ogg") -> str:
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
    return text if lang == "eng" else khaya_translate(text, f"{lang}-eng")


def from_english(text: str, lang: str) -> str:
    return text if lang == "eng" else khaya_translate(text, f"eng-{lang}")
