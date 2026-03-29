import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

DEFAULT_GENERATIVE_MODEL = "google/flan-t5-base"
DEFAULT_EXTRACTIVE_MODEL = "deepset/roberta-base-squad2"
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2500"))
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
USE_FAISS = os.getenv("USE_FAISS", "false").lower() == "true"

app = FastAPI(title="Document Q&A App", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


class AskResponse(BaseModel):
    answer: str
    contexts: list[str]
    sources_indexed: int
    retrieval_backend: str
    model_type: str


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    source_name: str


class DocumentStore:
    def __init__(self) -> None:
        self.records: list[ChunkRecord] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.faiss_index = None
        self.embedding_model = None
        self.backend = "memory"
        self._init_optional_faiss()

    def _init_optional_faiss(self) -> None:
        if not USE_FAISS:
            return

        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning("FAISS mode requested, but faiss-cpu or sentence-transformers is unavailable.")
            return

        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.backend = "faiss"
        logger.info("Initialized FAISS backend with embedding model %s", model_name)

    def add_document(self, source_name: str, text: str) -> int:
        chunks = split_text(text)
        start_index = len(self.records)
        for offset, chunk in enumerate(chunks):
            self.records.append(
                ChunkRecord(
                    chunk_id=f"{source_name}-{start_index + offset}",
                    text=chunk,
                    source_name=source_name,
                )
            )
        self._rebuild_index()
        return len(chunks)

    def _rebuild_index(self) -> None:
        texts = [record.text for record in self.records]
        if not texts:
            self.tfidf_matrix = None
            if self.faiss_index is not None:
                self._reset_faiss_index()
            return

        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        if self.faiss_index is not None and self.embedding_model is not None:
            self._reset_faiss_index()
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            self.faiss_index.add(embeddings)

    def _reset_faiss_index(self) -> None:
        if self.embedding_model is None or self.faiss_index is None:
            return

        import faiss  # type: ignore

        dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.faiss_index = faiss.IndexFlatL2(dimension)

    def search(self, question: str, top_k: int = TOP_K_CHUNKS) -> list[ChunkRecord]:
        if not self.records:
            return []

        top_k = max(1, min(top_k, len(self.records)))
        if self.backend == "faiss" and self.faiss_index is not None and self.embedding_model is not None:
            query_embedding = self.embedding_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
            _, indices = self.faiss_index.search(query_embedding, top_k)
            matches = [self.records[index] for index in indices[0] if index >= 0]
            return matches or self.records[:top_k]

        if self.tfidf_matrix is None:
            return self.records[:top_k]

        query_vector = self.vectorizer.transform([question])
        scores = (self.tfidf_matrix @ query_vector.T).toarray().ravel()
        ranked_indices = scores.argsort()[::-1][:top_k]
        matches = [self.records[index] for index in ranked_indices]
        return matches or self.records[:top_k]

    def stats(self) -> dict[str, Any]:
        return {
            "chunks_indexed": len(self.records),
            "retrieval_backend": self.backend,
        }


class QAService:
    def __init__(self) -> None:
        qa_mode = os.getenv("QA_MODEL_TYPE", "seq2seq").lower()
        if qa_mode == "extractive":
            self.model_type = "extractive"
            self.model_name = os.getenv("QA_MODEL_NAME", DEFAULT_EXTRACTIVE_MODEL)
        else:
            self.model_type = "seq2seq"
            self.model_name = os.getenv("QA_MODEL_NAME", DEFAULT_GENERATIVE_MODEL)
        self.pipe = None

    def ensure_loaded(self) -> None:
        if self.pipe is not None:
            return

        if self.model_type == "extractive":
            self.pipe = pipeline("question-answering", model=self.model_name, tokenizer=self.model_name)
        else:
            self.pipe = pipeline("text2text-generation", model=self.model_name, tokenizer=self.model_name)
        logger.info("Loaded QA model %s (%s)", self.model_name, self.model_type)

    def answer(self, question: str, contexts: list[str]) -> str:
        if not contexts:
            return "No documents are indexed yet. Upload a PDF or text file first."

        self.ensure_loaded()
        merged_context = "\n\n".join(contexts)
        merged_context = merged_context[:MAX_CONTEXT_CHARS]

        if self.model_type == "extractive":
            result = self.pipe(question=question, context=merged_context)
            answer = result["answer"].strip()
            return answer or "I could not find a confident answer in the uploaded document."

        prompt = (
            "Answer the question using only the context below. "
            "If the answer is not present, say you could not find it.\n\n"
            f"Context:\n{merged_context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        result = self.pipe(prompt, max_new_tokens=128, do_sample=False)
        answer = result[0]["generated_text"].strip()
        return answer or "I could not find a confident answer in the uploaded document."


def extract_text_from_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    upload.file.seek(0)

    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(upload.file.read())
            temp_path = temp_file.name
        try:
            reader = PdfReader(temp_path)
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages).strip()
        finally:
            Path(temp_path).unlink(missing_ok=True)
        if not text:
            raise HTTPException(status_code=400, detail="The PDF did not contain extractable text.")
        return text

    if suffix in {".txt", ".md", ".csv"}:
        return upload.file.read().decode("utf-8", errors="ignore").strip()

    raise HTTPException(status_code=400, detail="Unsupported file type. Upload a PDF or text file.")


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    clean_text = " ".join(text.split())
    if not clean_text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(clean_text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(clean_text[start:end])
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)
    return chunks


store = DocumentStore()
qa_service = QAService()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stats": store.stats(),
            "model_name": qa_service.model_name,
            "model_type": qa_service.model_type,
        },
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        **store.stats(),
        "model_name": qa_service.model_name,
        "model_loaded": qa_service.pipe is not None,
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    text = extract_text_from_upload(file)
    if not text:
        raise HTTPException(status_code=400, detail="No text could be extracted from the file.")

    chunk_count = store.add_document(file.filename, text)
    return {
        "message": "Document indexed successfully.",
        "filename": file.filename,
        "chunks_added": chunk_count,
        **store.stats(),
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    contexts = [record.text for record in store.search(question, payload.top_k or TOP_K_CHUNKS)]
    answer = qa_service.answer(question, contexts)
    return AskResponse(
        answer=answer,
        contexts=contexts,
        sources_indexed=len(store.records),
        retrieval_backend=store.backend,
        model_type=qa_service.model_type,
    )


@app.post("/ask-form", response_class=HTMLResponse)
async def ask_form(request: Request, question: str = Form(...)) -> HTMLResponse:
    contexts = [record.text for record in store.search(question, TOP_K_CHUNKS)]
    answer = qa_service.answer(question, contexts)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stats": store.stats(),
            "model_name": qa_service.model_name,
            "model_type": qa_service.model_type,
            "question": question,
            "answer": answer,
            "contexts": contexts,
        },
    )
