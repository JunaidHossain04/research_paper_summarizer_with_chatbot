from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from collections import defaultdict

from .summarizer import summarize_pdf_sections, chunk_text, answer_question
from .faiss_store import embed_chunks, create_faiss_index, search_faiss_index

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_DATA = {}

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_path = f"temp_{file.filename}"
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    summaries, sections = summarize_pdf_sections(pdf_path)
    # Chunk each section for embedding
    chunk_dict = defaultdict(list)
    for section, content in sections.items():
        for chunk in chunk_text(content):
            chunk_dict[section].append(chunk)
    chunk_titles, chunk_vectors = embed_chunks(chunk_dict)
    index = create_faiss_index(chunk_vectors)
    session_id = os.path.splitext(file.filename)[0]
    SESSION_DATA[session_id] = {
        "summaries": summaries,
        "chunk_dict": chunk_dict,
        "chunk_titles": chunk_titles,
        "index": index
    }
    os.remove(pdf_path)
    return {"session_id": session_id, "summaries": summaries}

@app.post("/ask/")
async def ask_question(session_id: str = Form(...), query: str = Form(...)):
    data = SESSION_DATA[session_id]
    results = search_faiss_index(
        data["index"], query, data["chunk_titles"], data["chunk_dict"]
    )
    # Combine text from top 4 chunks
    context = "\n\n".join([chunk for _, chunk in results])
    answer = answer_question(context, query)
    return {"answer": answer}
