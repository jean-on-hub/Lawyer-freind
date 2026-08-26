# Lawyer-freind

A free WhatsApp and Telegram chatbot that gives legal information to Ghanaians, powered by a RAG pipeline over Ghanaian legal PDFs.

> **Disclaimer:** This bot provides legal *information* only, not legal advice. For serious matters, users are directed to the Legal Aid Commission of Ghana (0302 975 749 · [lac.gov.gh](https://www.lac.gov.gh)).

---

## How It Works

1. Ghanaian legal PDFs are chunked and embedded into a FAISS vector store (`ghana_law_vectors/`).
2. When a user sends a message, the bot retrieves the most relevant legal text and passes it to an LLM.
3. Conversation history is tracked per user for multi-turn context.
4. If `GROQ_API_KEY` is set, the app uses **Groq** (cloud LLM). Otherwise it falls back to **Ollama** (local).

---

## Option A — Run Locally

### Prerequisites

| Tool | Purpose |
|---|---|
| Python 3.11 | Runtime |
| [Ollama](https://ollama.com) | Local LLM |
| [ngrok](https://ngrok.com) | Expose local server to Twilio/Telegram |
| Twilio account | WhatsApp messaging |

### Setup

```bash
git clone https://github.com/jean-on-hub/Lawyer-freind.git
cd Lawyer-freind
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

Pull the model and start Ollama:
```bash
ollama pull gemma4
ollama serve
```

Start the Flask app:
```bash
python scripts/server.py
```

Expose with ngrok (separate terminal):
```bash
ngrok http 5001
```

Set the Twilio webhook to `https://<ngrok-url>/whatsapp` and the Telegram webhook to `https://<ngrok-url>/telegram`.

---

## Option B — Deploy to AWS EC2 (Free Tier)

See **[AWS_DEPLOY.md](AWS_DEPLOY.md)** for the full step-by-step guide.

**Summary:**
1. Launch a t2.micro instance (free for 12 months) with Amazon Linux 2023
2. Install Docker, clone the repo, create `.env`
3. `docker compose up -d --build`
4. Set up nginx + Let's Encrypt for HTTPS
5. Point Twilio and Telegram webhooks at your domain

---

## CLI Bot (local testing — no WhatsApp/Telegram needed)

```bash
python scripts/bot.py
```

---

## User Commands

Start a new conversation (clears history):
```
new / reset / start over / new topic / clear / /new / /start
```

Switch language (see below):
```
english / twi / ga / ewe / dagbani / fante / frafra
```
Send `language` to see the current setting.

Choose how replies come back:
```
voice / text
```
Text is the default. The bot asks once, after a user's first voice note, rather than
prompting everyone — voice replies are metered and most users never want them.

---

## Voice Notes & Languages

**Voice notes work in English and Ghanaian Pidgin at no cost.** Audio goes to Groq's
Whisper (`whisper-large-v3-turbo`), which is on the same free key as the LLM —
2,000 requests/day, and Telegram's OGG/Opus is accepted directly, so no ffmpeg is needed.

**Ghanaian languages are metered.** Whisper does not support Twi, Ga, Ewe or Dagbani
(they are excluded for >50% word error rate), so those route through
[Ghana NLP's Khaya API](https://developer.khaya.ai) instead. Its free tier is
**100 calls per month**:

| Conversation | Khaya calls |
|---|---|
| English (text or voice) | **0** — entirely free |
| Twi/Ga/Ewe text | 2 (question in, answer out) |
| Twi/Ga/Ewe voice | 3 (transcribe + both translations) |

That is roughly **50 text or 33 voice** Ghanaian-language conversations per month.
The quota is tracked in `usage.db` and enforced before each call — when it runs out the
bot answers in English rather than erroring. Watch the remaining budget at `/stats`.

Set `KHAYA_API_KEY` in `.env` to enable; leave it unset to run English-only.
Language is chosen explicitly by the user rather than auto-detected, because detection
would spend a call on every message.

Translated replies include the **English original underneath**, so a bilingual reader
can catch a translation that has drifted.

> **Caution:** machine translation of legal information carries real risk — a
> mistranslated word about eviction or inheritance can lead someone to act wrongly.
> Keep answers short and always surface the Legal Aid Commission contact.

### Voice replies (Telegram only)

Opt-in per user via the `voice` command. Uses Groq's Orpheus TTS, which is **English
only**, allows roughly **100 free requests**, and **caps input at 200 characters** —
so the bot speaks a one-sentence summary and sends the full answer as text beside it.

An org admin must accept the model terms once at
[console.groq.com/playground](https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english),
otherwise every request returns HTTP 400. If TTS fails for any reason the answer still
goes out as text.

WhatsApp gets text only: Twilio requires a publicly hosted URL for outbound media,
which this bot has no way to serve.

---

## Usage Tracking

The bot logs every message to a local SQLite database (`usage.db`). View stats at:

```
https://YOUR_DOMAIN/stats?key=YOUR_STATS_KEY
```

Returns total messages, unique users, errors, and average response time per channel (WhatsApp / Telegram), plus a 30-day daily breakdown.

Set `STATS_KEY` in `.env` to protect the endpoint.

---

## Adding More Legal Documents

1. Drop new PDFs into `Legal_documents/`
2. Rebuild the index:
   ```bash
   python scripts/new_file_loader.py
   ```
3. Commit `ghana_law_vectors/` and redeploy

---

## Project Structure

```
Lawyer-freind/
├── Legal_documents/        # Ghanaian legal PDFs (source documents)
├── ghana_law_vectors/      # FAISS vector index (committed to git)
├── scripts/
│   ├── server.py           # Flask app — WhatsApp + Telegram webhooks
│   ├── bot.py              # Interactive CLI bot (local testing)
│   ├── new_file_loader.py  # PDF ingestion — run to rebuild index
│   └── download_model.py   # Pre-downloads embedding model (used in Docker build)
├── Dockerfile              # Docker image (Python 3.11, CPU-only torch)
├── docker-compose.yml      # EC2 deployment
├── start.sh                # Gunicorn entrypoint
├── AWS_DEPLOY.md           # Full AWS EC2 deployment guide
├── render.yaml             # Render deployment config (alternative)
├── railway.toml            # Railway deployment config (alternative)
├── .env.example            # Environment variable template
└── requirements.txt
```

---

## Legal Documents Included

| Document | Act |
|---|---|
| Ghana Constitution | — |
| Labour Act | 2003 |
| Land Act | 2020 |
| Rent Act | 1963 |
| Criminal Offences Act | — |
| Marriages Act | 1884–1985 |
| Children's Act | 1998 (Act 560) |
| Children's (Amendment) Act | 2016 |
| Legal Aid Commission Act | 2018 |
| Wills Act | — |
| High Court Civil Procedure Rules | CI 47 |
| Companies Act | 2019 (Act 992) |
| Intestate Succession Act | 1985 (PNDCL 111) |
| Domestic Violence Act | 2007 (Act 732) |
| Matrimonial Causes Act | 1971 (Act 367) |
| Criminal and Other Offences Procedure Act | 1960 (Act 30) |
| Contracts Act | 1960 (Act 25) |
| Road Traffic Act | 2004 (Act 683) |
| Data Protection Act | 2012 (Act 843) |
| Persons with Disability Act | 2006 (Act 715) |
| National Pensions Act | 2008 (Act 766) |
