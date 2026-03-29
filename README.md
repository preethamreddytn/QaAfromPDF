# Document Q&A Web App

A FastAPI application that:

- accepts PDF or text uploads
- extracts document text
- chunks and indexes content in memory
- optionally uses FAISS for vector retrieval
- answers questions with a Hugging Face MobileBERT extractive QA model
- includes a minimal browser UI

## Features

- `POST /upload` uploads and indexes a PDF or text file
- `POST /ask` answers questions using the indexed content
- `GET /` serves a simple HTML UI
- `GET /health` reports service status

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Optional FAISS setup

If you want vector retrieval with FAISS instead of the default lightweight in-memory retrieval:

```bash
pip install -r requirements-faiss.txt
set USE_FAISS=true
```

## API examples

Upload a document:

```bash
curl -X POST "http://127.0.0.1:8000/upload" -F "file=@sample.pdf"
```

Ask a question:

```bash
curl -X POST "http://127.0.0.1:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is the main topic?\"}"
```

## Configuration

Environment variables:

- `QA_MODEL_TYPE=lightweight-extractive` uses the built-in low-memory answerer
- `QA_MODEL_NAME=lightweight-extractive` keeps config aligned with the built-in answerer
- `USE_FAISS=true` enables FAISS retrieval if optional dependencies are installed
- `EMBEDDING_MODEL` sets the sentence-transformers model for FAISS mode
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K_CHUNKS`, `MAX_CONTEXT_CHARS` tune retrieval behavior

## Deployment

### Render

This repo includes [render.yaml](/C:/Users/preet/projects/CC/render.yaml) for Render deployment.

Recommended steps:

1. Push this folder to GitHub.
2. In Render, create a new Blueprint or Web Service from that repo.
3. Render will use:
   - build command: `pip install --upgrade pip && pip install -r requirements.txt`
   - start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. The Render config now uses csarron/mobilebert-uncased-squad-v2, a smaller MobileBERT QA model with a substantially lighter footprint than larger QA models.

### AWS App Runner / EC2 / ECS

Use this start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Set the service port to `8080` and install dependencies from `requirements.txt`.
If you want FAISS in the deployed service, also install `requirements-faiss.txt`.

### Google App Engine

Use the included `app.yaml`:

```bash
gcloud app deploy
```

## Notes

- The app stores chunks in process memory, so indexed documents reset on restart.
- FAISS is optional; if disabled or unavailable, the app falls back to the built-in in-memory retrieval path.
- The current default uses a smaller MobileBERT QA model to reduce deployment memory pressure.

