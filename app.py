from dotenv import load_dotenv
import requests
load_dotenv()  # take environment variables from .env


import os
import uuid
import io
import re
from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from docx import Document
from pypdf import PdfReader



#----------Flask Setup----------#

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

client = OpenAI()

ALLOWED_EXTENSIONS = {"pdf", "txt", "doc", "docx"}

#---------Helpers: files and URLs---------#

def allowed_file(filename: str) -> bool:

    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS



def extract_text_from_pdf(file_storage) -> str: 

    text = []
    reader = PdfReader(file_storage)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)


def extract_text_from_txt(file_storage) -> str:
    content = file_storage.read()

    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("Latin-1", errors="ignore")

def extract_text_from_docx(file_storage) ->str:

    file_bytes = file_storage.read()
    bio = io.BytesIO(file_bytes)
    doc = Document(bio)
    return "\n".join(p.text for p in doc.paragraphs)     

def extract_text_from_doc(file_storage) -> str:

    return "Support for doc not impliemnted yet convert to pdf or docx"

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



def tokenize(s: str):
    """Lowercase + basic word split; remove non-alpha chars."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return s.split()

def score_chunks(chunks, question: str, top_k: int = 5):
    

    q_tokens = set(tokenize(question))
    scored = []
    for ch in chunks:
        c_tokens = set(tokenize(ch["text"]))
        score = len(q_tokens & c_tokens)
        scored.append((score, ch))

    scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    
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
    for i, ch in enumerate(context_chunks, start = 1):
        source = ch["source"]
        snippet = ch["text"]
        context_parts.append(f"[Source {i} | {source}]\n{snippet}")

    context_block = "\n\n".join(context_parts) if context_parts else "No relevant context found."

    prompt = f"""
You are a helpful question-answering assistant.

You MUST:
- Answer ONLY using the information in the context.
- If the answer is not in the context, say you don't know.
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
    Calls OpenAI Responses API with a single text input.
    You can swap this out for Gemini or another provider if you want.
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    return response.output_text



#----------Flask Routes----------#

@app.before_request
def ensure_session():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    if "history" not in session: 
        session["history"] = []


        