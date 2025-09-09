# 🌌 Aurora AI — Advanced RAG + Speech AI Assistant

![Aurora AI Banner](/assets/Screenshot2025-09-10023455.png)

**Aurora AI** is an intelligent assistant that combines **Retrieval-Augmented Generation (RAG)** with **Speech AI** to provide fast, accurate insights from your documents and audio.  
Built with **Streamlit** for a beautiful interactive UI and **Supabase** for persistent storage, Aurora AI is designed for document understanding, audio transcription, semantic search, and conversation memory.

**Core idea:** enable a lightweight web app that *listens*, *reads*, *remembers*, and *answers* — backed by vector retrieval and modern generative models.

---

## 🚀 Highlights

- 🎙️ **Real-time audio recording** (browser) + transcription  
- 🔍 **Hybrid search** — semantic + keyword retrieval for precise answers  
- 🧾 **Document ingestion** with chunking and embeddings (RAG)  
- 🗣️ **Multi-speaker diarization** support (optional)  
- 🧭 **Contextual conversation memory** persisted in Supabase  
- 🕒 **Timeline & entity extraction** and exportable summaries  
- 🐳 Deployable via **Docker**, **Streamlit Cloud**, **Heroku**, **GCP**, or **AWS**

---

## 📸 UI Screenshots

> Add the screenshots you provided into `/assets/` and they will show here:

| Documents Upload | Recordings & Recorder | Quick Tools | Supabase Console |
|---|---:|:---|
| `/assets/Screenshot 2025-09-10 023353.png` | `/assets/Screenshot 2025-09-10 023407.png` | `/assets/Screenshot 2025-09-10 023428.png` | `/assets/Screenshot 2025-09-06 192519.png` |

---

## 🧩 Tech Stack

- **Frontend / App**: Streamlit (Python)  
- **Storage / DB**: Supabase (Postgres + Storage)  
- **Models / APIs**: Google Gemini (optional), Whisper (OpenAI or local), HuggingFace Transformers, Cohere (optional)  
- **Containerization**: Docker + Docker Compose  
- **Hosting**: Streamlit Cloud / Heroku / GCP / AWS

---

## 📦 Repository Layout (suggested)
```
/
├─ assets/                # screenshots, banner
├─ data/                  # sample data, test documents
├─ src/ or app/           # main Streamlit app and helpers
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .env.example
├─ DEPLOYMENT_GUIDE.md    # (kept for reference)
└─ README.md              # <- this file
```

---

## 🛠️ Quick Start (Local)

### 1. Clone
```bash
git clone https://github.com/Arzu0777/Aurora-AI.git
cd Aurora-AI
```

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create environment file
```bash
cp .env.example .env
# Edit .env and add API keys and Supabase credentials
```

**Important environment variables**
```
# Required / important
GOOGLE_API_KEY=
SUPABASE_URL=
SUPABASE_KEY=

# Optional
HUGGINGFACE_TOKEN=
COHERE_API_KEY=
WHISPER_MODEL=base
STREAMLIT_SERVER_PORT=8501
```

### 4. Run
```bash
streamlit run enhanced_rag_app_v2.py
# Open http://localhost:8501
```

---

## 🗄️ Supabase Database Schema

Use the following SQL to create the required tables in Supabase (Public schema):

```sql
-- Conversations table
create table if not exists public.conversations (
  id bigserial primary key,
  session_id text not null,
  query text not null,
  response text not null,
  context text,
  timestamp timestamptz default now()
);

-- Document metadata table
create table if not exists public.document_metadata (
  id bigserial primary key,
  filepath text not null,
  document_type text,
  chunk_count integer,
  processed_at timestamptz default now()
);

-- Give your anon/service roles permission
grant usage on schema public to anon, service_role;
grant all on public.conversations to anon, service_role;
grant all on public.document_metadata to anon, service_role;
```

> ⚠️ For production, lock down permissions and use row-level security (RLS) and role-based access instead of granting `all` to `anon` or `service_role`.

---

## 🐳 Docker (Recommended for reproducible deploys)

**Dockerfile (example)** — included in repo (or create):
```dockerfile
FROM python:3.9-slim
RUN apt-get update && apt-get install -y ffmpeg curl build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p temp outputs
EXPOSE 8501
CMD ["streamlit", "run", "enhanced_rag_app_v2.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**docker-compose.yml (example)**
```yaml
version: '3.8'
services:
  aurora:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./temp:/app/temp
      - ./outputs:/app/outputs
      - ./.env:/app/.env
    restart: unless-stopped
```

Run:
```bash
docker-compose up --build -d
```

---

## ⚡ Performance & Best Practices

- Limit conversation context window (e.g., last 8–12 messages) to reduce token usage.  
- Chunk large documents (CHUNK_SIZE ≈ 512–1000 tokens) and cap total chunks per doc.  
- Cache expensive model loads (Streamlit `@st.cache_data`) and I/O operations.  
- Use smaller Whisper model (`tiny` or `base`) for faster transcription on low-resource instances.  
- Use connection pooling for database access (e.g., `psycopg2.pool`).

Example caching:
```python
@st.cache_data
def load_whisper_model(model_name="base"):
    return whisper.load_model(model_name)
```

---

## 🐛 Troubleshooting

- **Streamlit not starting** — ensure port is free; try `--server.port=8502`.  
- **Audio issues** — verify `ffmpeg` is installed and accessible.  
- **Supabase connection errors** — check `SUPABASE_URL` and `SUPABASE_KEY`.  
- **Model OOM / memory errors** — use smaller models or increase instance RAM.

---

## 🔒 Security & Production Notes

- Never commit `.env` or API keys to Git. Use platform secrets.  
- For public deployments, enable HTTPS and configure CORS and XSRF protection.  
- Implement authentication (Supabase Auth or OAuth) before exposing features that modify data.  
- Add rate-limiting and monitoring for model/API usage.

---

## ♻️ Backup & Monitoring

- Schedule DB backups (`pg_dump`) and store artifacts in cloud storage.  
- Add basic performance metrics to the sidebar (processing time, memory).  
- Log user activity (with privacy in mind) for debugging.

---

## 🛣️ Roadmap & Ideas

- Add user authentication & RBAC (Supabase Auth)  
- Integrate advanced RAG retrievers (FAISS/Chroma) for local vector storage  
- Multi-language support for transcription and search  
- A/B tests between different LLM providers for cost/perf tradeoffs

---

## 🙌 Contributing

Welcome contributions. Please:
1. Fork the repo
2. Create a feature branch (`feat/…`)
3. Open a PR with a clear description & tests (if applicable)

---

## 📬 Credits & Contact

Developed by **Arzu** — (GitHub: `Arzu0777`)  
If you want me to also create release-ready assets (banner, compressed screenshots, or a GH Actions workflow), I can add them to the repo.

**Enjoy building — Aurora AI brings memory to assistants.** 🌌
