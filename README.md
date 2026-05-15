# Lawyer-freind

A free WhatsApp chatbot that gives legal information to Ghanaians, powered by a RAG pipeline over Ghanaian legal documents. Runs locally with Ollama or in the cloud with Groq — same codebase, no changes needed.

> **Disclaimer:** This bot provides legal *information* only, not legal advice. For serious matters, users are directed to the Legal Aid Commission of Ghana (0800-100-950).

---

## How It Works

1. Ghanaian legal PDFs are chunked and embedded into a FAISS vector store (`ghana_law_vectors/`).
2. When a user sends a WhatsApp message via Twilio, the Flask app retrieves the most relevant legal text chunks and passes them to an LLM to generate an answer.
3. Conversation history is tracked per user (by phone number) for multi-turn context.
4. If `GROQ_API_KEY` is set, the app uses **Groq** (cloud). Otherwise it falls back to **Ollama** (local).

---

## Option A — Run Locally

### Prerequisites

| Tool | Purpose |
|---|---|
| Python 3.10+ | Runtime |
| [Ollama](https://ollama.com) | Local LLM server |
| [ngrok](https://ngrok.com) | Expose local server to Twilio |
| Twilio account | WhatsApp messaging API |

### 1. Clone & install

```bash
git clone https://github.com/jean-on-hub/Lawyer-freind.git
cd Lawyer-freind
pip install -r requirements.txt
```

### 2. Pull the model and start Ollama

```bash
ollama pull gemma4
ollama serve
```

### 3. Build the vector index

```bash
python scripts/new_file_loader.py
```

### 4. Start the Flask app

```bash
python scripts/app.py
```

### 5. Expose with ngrok (in a separate terminal)

```bash
ngrok http 5001
```

### 6. Configure Twilio

1. Go to [Twilio Console](https://console.twilio.com) → Messaging → Try it out → Send a WhatsApp message.
2. Under **Sandbox Settings**, set the **When a message comes in** URL to:
   ```
   https://<your-ngrok-url>/whatsapp
   ```

---

## Option B — Deploy to the Cloud (Render + Groq)

No local machine required. Fully free.

### Prerequisites

| Service | Purpose | Cost |
|---|---|---|
| [Render](https://render.com) | Hosts the Flask app | Free tier |
| [Groq](https://console.groq.com) | Cloud LLM API | Free tier |
| [Twilio](https://console.twilio.com) | WhatsApp messaging | Free sandbox |
| GitHub | Source for auto-deploy | Free |

### 1. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) → API Keys → Create key.

### 2. Push the repo to GitHub

Make sure `ghana_law_vectors/` (the FAISS index) is committed — Render needs it at startup.

```bash
git add .
git commit -m "deploy"
git push
```

### 3. Deploy on Render

1. Go to [render.com](https://render.com) → **New** → **Web Service**
2. Connect your GitHub repo
3. Render will detect `render.yaml` automatically — confirm the settings
4. Under **Environment Variables**, add:
   - `GROQ_API_KEY` → your key from step 1
5. Click **Deploy**

Render gives you a public URL like `https://lawyer-freind.onrender.com`.

### 4. Configure Twilio

Set the webhook URL to:
```
https://lawyer-freind.onrender.com/whatsapp
```

> **Note:** Render's free tier spins down after 15 minutes of inactivity. The first message after idle will take ~30 seconds (cold start). Upgrade to Render's $7/month plan to keep it always-on.

---

## CLI Bot (local testing — no WhatsApp needed)

```bash
python scripts/bot.py
```

---

## Adding More Legal Documents

1. Drop new PDFs into `Legal_documents/`
2. Rebuild the index:
   ```bash
   python scripts/new_file_loader.py
   ```
3. Commit `ghana_law_vectors/` and redeploy (or restart locally)

---

## Project Structure

```
Lawyer-freind/
├── Legal_documents/        # Ghanaian legal PDFs (source documents)
├── ghana_law_vectors/      # FAISS vector index (commit this to GitHub)
├── scripts/
│   ├── app.py              # Flask webhook — auto-selects Groq or Ollama
│   ├── bot.py              # Interactive CLI bot (local testing)
│   ├── new_file_loader.py  # PDF ingestion — run to rebuild index
│   ├── main.py             # Legacy ingestion script
│   └── echo.py             # Minimal echo webhook (Twilio testing only)
├── Procfile                # Gunicorn start command for Render
├── render.yaml             # Render deployment config
├── .env.example            # Environment variable template
└── requirements.txt
```

---

## Legal Documents Included

- Ghana Constitution
- Labour Act 2003
- Land Act 2020
- Rent Act 1963
- Criminal Offences Act
- Marriages Act 1884–1985
- Children's Act 1998 (Act 560)
- Children's (Amendment) Act 2016
- Legal Aid Commission Act 2018
- Wills Act
- High Court (Civil Procedure) Rules 2004 (CI 47)
- Companies Act 2019 (Act 992)
- Intestate Succession Act 1985 (PNDCL 111)
- Domestic Violence Act 2007 (Act 732)
- Matrimonial Causes Act 1971 (Act 367)
- Criminal and Other Offences (Procedure) Act 1960 (Act 30)
- Contracts Act 1960 (Act 25)
- Road Traffic Act 2004 (Act 683)
- Data Protection Act 2012 (Act 843)
- Persons with Disability Act 2006 (Act 715)
- National Pensions Act 2008 (Act 766)
