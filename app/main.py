import json
import logging
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pypdf import PdfReader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / 'templates'
STATIC_DIR = BASE_DIR / 'static'


def load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env(BASE_DIR.parent / '.env')

DEFAULT_GEMINI_MODEL = os.getenv('QA_MODEL_NAME', 'gemini-2.5-flash')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_API_BASE = os.getenv('GEMINI_API_BASE', 'https://generativelanguage.googleapis.com/v1beta/models')
MAX_CONTEXT_CHARS = int(os.getenv('MAX_CONTEXT_CHARS', '1600'))
TOP_K_CHUNKS = int(os.getenv('TOP_K_CHUNKS', '3'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '700'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '120'))
MAX_SENTENCES_TO_SCAN = int(os.getenv('MAX_SENTENCES_TO_SCAN', '24'))

TOKEN_PATTERN = re.compile(r'\b[a-zA-Z0-9]+\b')
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')
YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
NUMBER_PATTERN = re.compile(r'\b\d+(?:\.\d+)?\b')
PROPER_NOUN_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'how',
    'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was', 'what',
    'when', 'where', 'which', 'who', 'why', 'with'
}
QUESTION_PREFIXES = ('who', 'what', 'when', 'where', 'why', 'how', 'which')

app = FastAPI(title='Document Q&A App', version='1.0.0')
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
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
    term_counts: Counter[str]
    term_total: int


@dataclass
class SentenceCandidate:
    text: str
    source_name: str
    chunk_id: str
    score: float


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token and token not in STOP_WORDS and not token.isdigit()
    ]


def split_sentences(text: str) -> list[str]:
    compact = ' '.join(text.split())
    if not compact:
        return []
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(compact) if sentence.strip()]
    return sentences or [compact]


def clean_question(question: str) -> str:
    question = ' '.join(question.split()).strip()
    return question[:-1] if question.endswith('?') else question


def question_type(question: str) -> str:
    lowered = clean_question(question).lower()
    for prefix in QUESTION_PREFIXES:
        if lowered.startswith(prefix + ' ') or lowered == prefix:
            return prefix
    return 'generic'


class DocumentStore:
    def __init__(self) -> None:
        self.records: list[ChunkRecord] = []
        self.document_frequencies: Counter[str] = Counter()
        self.total_documents = 0
        self.backend = 'memory'

    def add_document(self, source_name: str, text: str) -> int:
        chunks = split_text(text)
        start_index = len(self.records)
        for offset, chunk in enumerate(chunks):
            term_counts = Counter(tokenize(chunk))
            self.records.append(
                ChunkRecord(
                    chunk_id=f'{source_name}-{start_index + offset}',
                    text=chunk,
                    source_name=source_name,
                    term_counts=term_counts,
                    term_total=sum(term_counts.values()),
                )
            )
        self._rebuild_index()
        return len(chunks)

    def _rebuild_index(self) -> None:
        self.total_documents = len(self.records)
        self.document_frequencies = Counter()
        for record in self.records:
            self.document_frequencies.update(record.term_counts.keys())

    def _idf(self, term: str) -> float:
        frequency = self.document_frequencies.get(term, 0)
        return math.log((1 + self.total_documents) / (1 + frequency)) + 1.0

    def _score_record(self, query_terms: Counter[str], record: ChunkRecord) -> float:
        if not query_terms:
            return 0.0

        overlap = set(query_terms) & set(record.term_counts)
        if not overlap:
            return 0.0

        score = 0.0
        for term in overlap:
            normalized_tf = record.term_counts[term] / max(record.term_total, 1)
            score += query_terms[term] * normalized_tf * self._idf(term)

        score += 0.2 * len(overlap)
        return score

    def search(self, question: str, top_k: int = TOP_K_CHUNKS) -> list[ChunkRecord]:
        if not self.records:
            return []

        top_k = max(1, min(top_k, len(self.records)))
        query_terms = Counter(tokenize(question))
        if not query_terms:
            return self.records[:top_k]

        scored_records = [
            (self._score_record(query_terms, record), index, record)
            for index, record in enumerate(self.records)
        ]
        scored_records.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        matches = [record for score, _, record in scored_records[:top_k] if score > 0]
        return matches or self.records[:top_k]

    def stats(self) -> dict[str, Any]:
        return {
            'chunks_indexed': len(self.records),
            'retrieval_backend': self.backend,
        }


class LocalExtractiveFallback:
    def _question_bonus(self, q_type: str, sentence: str) -> float:
        lowered = sentence.lower()
        bonus = 0.0
        if q_type == 'when' and (YEAR_PATTERN.search(sentence) or NUMBER_PATTERN.search(sentence)):
            bonus += 0.8
        if q_type == 'who' and PROPER_NOUN_PATTERN.search(sentence):
            bonus += 0.6
        if q_type == 'where' and any(word in lowered for word in (' in ', ' at ', ' near ', ' from ', ' to ', ' located ', ' based ')):
            bonus += 0.5
        if q_type == 'how' and NUMBER_PATTERN.search(sentence):
            bonus += 0.5
        if q_type == 'why' and any(word in lowered for word in (' because ', ' due to ', ' since ', ' so that ', ' reason ')):
            bonus += 0.6
        return bonus

    def _sentence_score(self, question: str, sentence: str, store: DocumentStore) -> float:
        query_terms = Counter(tokenize(question))
        sentence_terms = Counter(tokenize(sentence))
        if not query_terms or not sentence_terms:
            return 0.0

        overlap = set(query_terms) & set(sentence_terms)
        if not overlap:
            return 0.0

        score = 0.0
        for term in overlap:
            normalized_tf = sentence_terms[term] / max(sum(sentence_terms.values()), 1)
            score += query_terms[term] * normalized_tf * store._idf(term)

        lowered_question = clean_question(question).lower()
        lowered_sentence = sentence.lower()
        if lowered_question in lowered_sentence:
            score += 1.0

        score += 0.25 * len(overlap)
        score += self._question_bonus(question_type(question), sentence)
        score -= 0.0015 * len(sentence)
        return score

    def answer(self, question: str, records: list[ChunkRecord], store: DocumentStore) -> tuple[str, list[str]]:
        if not records:
            return 'No documents are indexed yet. Upload a PDF or text file first.', []

        candidates: list[SentenceCandidate] = []
        contexts: list[str] = []
        for record in records:
            contexts.append(record.text[:MAX_CONTEXT_CHARS])
            for sentence in split_sentences(record.text):
                candidates.append(
                    SentenceCandidate(
                        text=sentence,
                        source_name=record.source_name,
                        chunk_id=record.chunk_id,
                        score=self._sentence_score(question, sentence, store),
                    )
                )

        candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = candidates[:MAX_SENTENCES_TO_SCAN]
        if not candidates or candidates[0].score <= 0:
            return 'I could not find a confident answer in the uploaded document.', contexts

        best = candidates[0]
        answer_parts = [best.text]
        if len(candidates) > 1:
            second = candidates[1]
            if second.chunk_id == best.chunk_id and second.score >= best.score * 0.82 and second.text != best.text:
                answer_parts.append(second.text)

        return ' '.join(answer_parts)[:MAX_CONTEXT_CHARS], contexts


class QAService:
    def __init__(self) -> None:
        self.model_name = DEFAULT_GEMINI_MODEL
        self.fallback = LocalExtractiveFallback()

    def _build_prompt(self, question: str, contexts: list[str]) -> str:
        merged_context = '\n\n'.join(contexts)[:MAX_CONTEXT_CHARS]
        return (
            'Answer the question using only the context below. '
            'If the answer is not present, say you could not find it. '
            'Respond in plain English with only the answer.\n\n'
            f'Context:\n{merged_context}\n\n'
            f'Question: {question}'
        )

    def _parse_gemini_response(self, payload: dict[str, Any]) -> str:
        candidates = payload.get('candidates') or []
        for candidate in candidates:
            content = candidate.get('content') or {}
            for part in content.get('parts') or []:
                text = part.get('text')
                if text:
                    return str(text).strip()
        return ''

    def _call_gemini(self, prompt: str) -> str:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail='GEMINI_API_KEY is not configured on the server.')

        model_path = self.model_name.removeprefix('models/')
        api_url = f"{GEMINI_API_BASE.rstrip('/')}/{model_path}:generateContent?key={GEMINI_API_KEY}"
        body = json.dumps(
            {
                'contents': [
                    {
                        'parts': [
                            {'text': prompt}
                        ]
                    }
                ],
                'generationConfig': {
                    'temperature': 0.2,
                    'maxOutputTokens': 160,
                },
            }
        ).encode('utf-8')
        req = request.Request(
            api_url,
            data=body,
            headers={
                'Content-Type': 'application/json',
            },
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=60) as response:
                raw = response.read().decode('utf-8')
        except error.HTTPError as exc:
            details = exc.read().decode('utf-8', errors='ignore')
            logger.exception('Gemini API error: %s', details)
            message = details.strip() or f'Gemini API request failed (status={exc.code}).'
            raise HTTPException(status_code=502, detail=message) from exc
        except error.URLError as exc:
            logger.exception('Gemini API network error')
            raise HTTPException(status_code=502, detail='Could not reach the Gemini API.') from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=502, detail='Invalid response from Gemini API.') from exc

        if isinstance(parsed, dict) and parsed.get('error'):
            raise HTTPException(status_code=502, detail=json.dumps(parsed['error']))

        text = self._parse_gemini_response(parsed)
        if not text:
            raise HTTPException(status_code=502, detail='Unexpected response from Gemini API.')
        return text

    def answer(self, question: str, records: list[ChunkRecord], store: DocumentStore) -> tuple[str, list[str], str]:
        fallback_answer, contexts = self.fallback.answer(question, records, store)
        if not records:
            return fallback_answer, contexts, 'local-extractive-v2'

        if not GEMINI_API_KEY:
            return fallback_answer, contexts, 'local-extractive-v2'

        prompt = self._build_prompt(question, contexts)
        answer = self._call_gemini(prompt)
        return (answer or fallback_answer), contexts, self.model_name


def extract_text_from_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or '').suffix.lower()
    upload.file.seek(0)

    if suffix == '.pdf':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(upload.file.read())
            temp_path = temp_file.name
        try:
            reader = PdfReader(temp_path)
            pages = [page.extract_text() or '' for page in reader.pages]
            text = '\n'.join(pages).strip()
        finally:
            Path(temp_path).unlink(missing_ok=True)
        if not text:
            raise HTTPException(status_code=400, detail='The PDF did not contain extractable text.')
        return text

    if suffix in {'.txt', '.md', '.csv'}:
        return upload.file.read().decode('utf-8', errors='ignore').strip()

    raise HTTPException(status_code=400, detail='Unsupported file type. Upload a PDF or text file.')


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    clean_text = ' '.join(text.split())
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


@app.get('/', response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        'index.html',
        {
            'request': request,
            'stats': store.stats(),
            'model_name': qa_service.model_name,
            'model_type': 'gemini-api' if GEMINI_API_KEY else 'local-extractive-v2',
        },
    )


@app.get('/health')
async def health() -> dict[str, Any]:
    return {
        'status': 'ok',
        **store.stats(),
        'model_name': qa_service.model_name,
        'model_loaded': bool(GEMINI_API_KEY),
    }


@app.post('/upload')
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail='Missing filename.')

    text = extract_text_from_upload(file)
    if not text:
        raise HTTPException(status_code=400, detail='No text could be extracted from the file.')

    chunk_count = store.add_document(file.filename, text)
    return {
        'message': 'Document indexed successfully.',
        'filename': file.filename,
        'chunks_added': chunk_count,
        **store.stats(),
    }


@app.post('/ask', response_model=AskResponse)
async def ask_question(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail='Question cannot be empty.')

    records = store.search(question, payload.top_k or TOP_K_CHUNKS)
    answer, contexts, answer_source = qa_service.answer(question, records, store)
    return AskResponse(
        answer=answer,
        contexts=contexts,
        sources_indexed=len(store.records),
        retrieval_backend=store.backend,
        model_type=answer_source,
    )


@app.post('/ask-form', response_class=HTMLResponse)
async def ask_form(request: Request, question: str = Form(...)) -> HTMLResponse:
    records = store.search(question, TOP_K_CHUNKS)
    answer, contexts, answer_source = qa_service.answer(question, records, store)
    return templates.TemplateResponse(
        'index.html',
        {
            'request': request,
            'stats': store.stats(),
            'model_name': qa_service.model_name,
            'model_type': answer_source,
            'question': question,
            'answer': answer,
            'contexts': contexts,
        },
    )



