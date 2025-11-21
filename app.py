from dotenv import load_dotenv
import requests
load_dotenv()  # take environment variables from .env


import csv
import os
import uuid
import io
import re
import sqlite3
from pathlib import Path
import numpy as np

import faiss
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from flask import Flask, render_template, request, session, Response
from werkzeug.utils import secure_filename

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from docx import Document
from pypdf import PdfReader

# Optional: point pytesseract at a specific binary via env var.
TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD



#----------Flask Setup----------#

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "articles.db"


ALLOWED_EXTENSIONS = {"pdf", "txt", "doc", "docx", "jpg", "jpeg", "png"}

#----------Storage (SQLite)----------#

def init_db():
    DATA_DIR.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                url TEXT,
                content TEXT,
                fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def store_article(source: str, url: str, content: str):
    """Persist a copy of fetched/processed content for auditing or reuse."""
    if not content:
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO articles (source, url, content) VALUES (?, ?, ?)",
            (source, url, content),
        )


def list_articles(limit: int = 50):
    """Return recent articles with metadata only."""
    limit = max(1, min(limit, 500))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, source, url, fetched_at, length(content) as content_length
            FROM articles
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_article(article_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, source, url, fetched_at, content FROM articles WHERE id = ?",
            (article_id,),
        ).fetchone()
        return dict(row) if row else None


def get_articles_by_ids(ids):
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    query = f"""
        SELECT id, source, url, fetched_at, content
        FROM articles
        WHERE id IN ({placeholders})
        ORDER BY id DESC
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, ids).fetchall()
        return [dict(r) for r in rows]


# Initialize storage on import so it's ready for requests.
init_db()

#---------Helpers: files and URLs---------#

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def extract_text_from_pdf(file_storage) -> str:
    # Read bytes to allow OCR fallback on scanned PDFs.
    file_storage.seek(0)
    pdf_bytes = file_storage.read()

    text = []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    except Exception:
        text = []

    cleaned = clean_text("\n".join(text))
    if cleaned:
        return cleaned

    # OCR fallback for scanned PDFs
    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        ocr_texts = []
        for img in images:
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text:
                ocr_texts.append(ocr_text)
        ocr_cleaned = clean_text("\n".join(ocr_texts))
        return ocr_cleaned or "No text extracted from scanned PDF."
    except Exception:
        return "No text extracted from scanned PDF."


def extract_text_from_txt(file_storage) -> str:
    content = file_storage.read()
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("Latin-1", errors="ignore")


def extract_text_from_docx(file_storage) -> str:
    file_bytes = file_storage.read()
    bio = io.BytesIO(file_bytes)
    doc = Document(bio)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text_from_doc(file_storage) -> str:
    return "Support for doc not impliemnted yet convert to pdf or docx"


def extract_text_from_image(file_storage) -> str:
    try:
        image = Image.open(file_storage)
        text = pytesseract.image_to_string(image)
        cleaned = clean_text(text)
        return cleaned or "No text extracted from image."
    except Exception:
        return "No text extracted from image."


def extract_text_from_text(file_storage) -> str:
    ext = file_storage.filename.rsplit(".", 1)[1].lower()

    if ext == "pdf":
        return extract_text_from_pdf(file_storage)
    elif ext == "docx":
        return extract_text_from_docx(file_storage)
    elif ext == "txt":
        return extract_text_from_txt(file_storage)
    elif ext == "doc":
        return extract_text_from_doc(file_storage)
    elif ext in {"png", "jpg", "jpeg"}:
        return extract_text_from_image(file_storage)
    else:
        return ""  # Unsupported file type


def fetch_text_from_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"Error fetching URL: {e}"
    
    soup = BeautifulSoup(resp.text, "html.parser")

    # remove scripts and styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # basic cleanup

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    cleaned = "\n".join(lines)
    # Store a copy of the fetched article for later reuse/auditing.
    store_article("url", url, cleaned)
    return cleaned

#----Text Chunking  and relevance----#
def chunk_text(text: str, source_id: str, chunk_size: int = 800, overlap: int = 120):
    """
    Very simple character-based chunking with overlap.
    Returns list of dicts: {"source": ..., "text": ...}
    """
       
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append({"source" : source_id, "text" : chunk})
        if end == length:
            break
        start = end - overlap
    return chunks


def embed_texts(texts):
    """Return embeddings as float32 numpy array."""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vectors = [d.embedding for d in resp.data]
    return np.array(vectors, dtype="float32")


def tokenize(s: str):
    """Lowercase + basic word split; remove non-alpha chars."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return s.split()

def score_chunks(chunks, question: str, top_k: int = 5):
    """Retrieve top chunks using embeddings + FAISS, with keyword fallback."""
    if not chunks or not question:
        return []
    try:
        texts = [ch["text"] for ch in chunks]
        chunk_vecs = embed_texts(texts)
        if chunk_vecs.size == 0:
            return []
        faiss.normalize_L2(chunk_vecs)
        dim = chunk_vecs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(chunk_vecs)

        q_vec = embed_texts([question])
        faiss.normalize_L2(q_vec)
        k = min(top_k, len(chunks))
        _, idx = index.search(q_vec, k)

        selected = []
        for i in idx[0]:
            if i == -1:
                continue
            selected.append(chunks[i])
        return selected
    except Exception:
        # fallback to simple keyword overlap if embeddings fail
        q_tokens = set(tokenize(question))
        scored = []
        for ch in chunks:
            c_tokens = set(tokenize(ch["text"]))
            score = len(q_tokens & c_tokens)
            scored.append((score, ch))

        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return chunks[:top_k]

        return [ch for score, ch in scored[:top_k]]

#----------LLM Calls----------#



def build_prompt(question: str, context_chunks, history):

    """
    Build a single string prompt including:
    - instructions
    - conversation history
    - selected context
    - new question
    """

    history_lines = []
    for msg in history:
        role = msg.get("role", "user")
        prefix = "User" if role == "user" else "Assistant"
        history_lines.append(f"{prefix}: {msg['content']}")

    history_block = "\n".join(history_lines) if history_lines else "No prior conversation."


    # build context with source labels

    context_parts = []
    for i, ch in enumerate(context_chunks, start=1):
        source = ch["source"]
        snippet = ch["text"]
        context_parts.append(f"[Source {i} | {source}]\n{snippet}")

    context_block = "\n\n".join(context_parts) if context_parts else "No relevant context found."

    # Prefer context when it answers the question; otherwise allow general knowledge so we don't get stuck on irrelevant snippets.
    constraints = (
        "Prefer the provided context. If it answers the question, cite sources. If it does not fully answer, use your own knowledge and say you are answering from general knowledge."
        if context_parts
        else "Use your own knowledge to answer since no relevant context was supplied."
    )

    prompt = f"""
You are a helpful question-answering assistant.

You MUST:
- {constraints}
- When possible, refer to the sources by their labels like [Source 1].

Conversation so far:
{history_block}

Context from user's documents and URLs:
{context_block}

New user question:
{question}

Now provide a clear, concise answer labeled as 'Answer:'.
If relevant, reference sources in-line, e.g. "According to [Source 2]".
"""
    return prompt.strip()


def call_llm(prompt: str):

    """
    Calls OpenAI Chat Completions API with a single text input.
    You can swap this out for Gemini or another provider if you want.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        timeout=60  # 60 second timeout
    )

    return response.choices[0].message.content



#----------Flask Routes----------#

@app.before_request
def ensure_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    if "history" not in session: 
        session["history"] = []

@app.route("/", methods = ["GET", "POST"])

def index():
    answer  = None 
    used_chunks = []
    filenames = []
    url_list = []
    stored_articles = list_articles(limit=50)
    selected_ids = []

    if request.method == "POST":
        # Allow clearing conversation/history without calling the LLM.
        if request.form.get("clear") == "1":
            session["history"] = []
            return render_template(
                "index.html",
                history=[],
                answer=None,
                used_chunks=[],
                filenames=[],
                urls=[],
                stored_articles=stored_articles,
                selected_ids=[],
            )

        question = request.form.get("question", "").strip()
        url_text = request.form.get("urls", "").strip()
        selected_ids = request.form.getlist("stored_ids")
        if len(selected_ids) > 10:
            selected_ids = selected_ids[:10]
        selected_ids_int = [int(x) for x in selected_ids if x.isdigit()]


        if not question:
            question = "(empty question)"


        # ----- Gather sources: files ---------

        uploaded_files = request.files.getlist("files")
        all_chunks = []


        for f in uploaded_files:
            if f and allowed_file(f.filename):
                filename = secure_filename(f.filename)
                # NOTE: for some extractors we consume file, so keep a copy per file
                file_bytes = f.read()
                f_stream = io.BytesIO(file_bytes)
                f_stream.name = filename

                if filename.lower().endswith(".pdf"):
                    text = extract_text_from_pdf(f_stream)
                elif filename.lower().endswith(".txt"):
                    text = file_bytes.decode("utf-8", errors="ignore")
                elif filename.lower().endswith(".docx"):
                    text = extract_text_from_docx(io.BytesIO(file_bytes))
                elif filename.lower().endswith(".doc"):
                    text = extract_text_from_doc(io.BytesIO(file_bytes))
                else:
                    text = ""

                if text:
                    store_article("file", filename, text)

                source_id = f"file: {filename}"
                filenames.append(source_id)
                all_chunks.extend(chunk_text(text, source_id))

        # ----- Gather sources: stored articles ---------
        if selected_ids_int:
            stored_rows = get_articles_by_ids(selected_ids_int)
            for row in stored_rows:
                source_id = f"stored #{row['id']} ({row.get('url') or row.get('source')})"
                all_chunks.extend(chunk_text(row.get("content", ""), source_id))

        # ----- Gather sources: URLs ---------
        if url_text: 

            raw_urls = [u.strip() for u in url_text.splitlines() if u.strip()]
            for url in raw_urls:
                url_list.append(url)
                text = fetch_text_from_url(url)
                source_id = f"url: {url}"
                all_chunks.extend(chunk_text(text, source_id))
        # ----- Select relevant chunks ---------
        relevant_chunks = score_chunks(all_chunks, question, top_k=5)

        # ----- Build prompt using history ---------
        history = session.get("history", [])
        prompt = build_prompt(question, relevant_chunks, history)

        # ----- Call LLM ---------
        try:
            answer = call_llm(prompt)
        except Exception as e:
            answer = f"Error calling LLM API: {e}"

        used_chunks = relevant_chunks


        # ----- Update Converstioanl history ---------
        history.append({"role": "user", "content": question})
        if answer:
            history.append({"role": "assistant", "content": answer})

        # limit to last 10 messages to keep session small 
        if len(history) > 10:
            history = history[-10:]
        session["history"] = history



    return render_template(
        "index.html",
        history = session.get("history", []),
        answer = answer,
        used_chunks = used_chunks,
        filenames=filenames,
        urls=url_list,
        stored_articles=stored_articles,
        selected_ids=selected_ids,
    )

@app.route("/batch", methods=["GET", "POST"])
def batch():
    results = []
    limit_note = None

    if request.method == "POST":
        url_text = request.form.get("urls", "").strip()
        raw_urls = [u.strip() for u in url_text.splitlines() if u.strip()]
        if len(raw_urls) > 10:
            limit_note = "Only the first 10 URLs were processed."
            raw_urls = raw_urls[:10]

        for url in raw_urls:
            text = fetch_text_from_url(url)
            # fetch_text_from_url already stores the article; we just report outcome
            ok = not text.startswith("Error fetching URL")
            results.append({"url": url, "status": "Saved" if ok else text[:120] + ("..." if len(text) > 120 else "")})

    return render_template(
        "batch.html",
        results=results,
        limit_note=limit_note,
    )

@app.route("/stored")
def stored():
    articles = list_articles(limit=200)
    article_id = request.args.get("id", type=int)
    detail = get_article(article_id) if article_id else None
    return render_template(
        "stored.html",
        articles=articles,
        detail=detail,
    )


@app.route("/stored/export")
def stored_export():
    articles = list_articles(limit=1000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "source", "url", "fetched_at", "content_length"])
    for a in articles:
        writer.writerow([
            a.get("id", ""),
            a.get("source", ""),
            a.get("url", ""),
            a.get("fetched_at", ""),
            a.get("content_length", ""),
        ])
    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=stored_articles.csv"},
    )

if __name__ == "__main__":
    app.run(debug=True)
    


                



        
