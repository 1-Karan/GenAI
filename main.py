from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.concurrency import run_in_threadpool
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import chromadb
import os
import uuid
import pdfplumber
from docx import Document  
import numpy as np

app = FastAPI()

# Initialize ChromaDB client and create or retrieve collection for storing documents
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("documents")

# Load the Sentence Transformer model 
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

os.makedirs("./temp_files", exist_ok=True)

def read_pdf(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        return "".join([page.extract_text() for page in pdf.pages])

def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Utility function to extract text from different document types
async def extract_text(file: UploadFile) -> str:
    temp_path = f"./temp_files/{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Extract text based on document type
        if file.content_type == "application/pdf":
            text = read_pdf(temp_path)
        elif file.content_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            text = read_docx(temp_path)
        elif file.content_type == "text/plain":
            with open(temp_path, "r", encoding="utf-8") as txt_file:
                text = txt_file.read()
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    finally:
        os.remove(temp_path)
    
    return text

# Endpoint to ingest documents into ChromaDB
@app.post("/ingest/")
async def ingest_document(file: UploadFile = File(...)):
    text = await extract_text(file)
    sections = [section.strip() for section in text.split("\n") if section.strip()]
    
    # Encode each section in a separate thread pool to prevent blocking
    embeddings = await run_in_threadpool(lambda: [embedding_model.encode(section) for section in sections])
    ids = [str(uuid.uuid4()) for _ in sections]

    # Add each section to ChromaDB
    for i, (embedding, section) in enumerate(zip(embeddings, sections)):
        collection.add(
            embeddings=[embedding],
            documents=[section],
            metadatas=[{"section": f"Section {i+1}", "type": file.content_type}],
            ids=[ids[i]]
        )
    
    return {"message": "Document ingested successfully"}

# Data model for query request
class QueryRequest(BaseModel):
    query: str

# Endpoint to query documents based on a user-provided query string
@app.post("/query/")
async def query_documents(request: QueryRequest):
    query_embedding = await run_in_threadpool(lambda: embedding_model.encode(request.query))
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    
    # Apply a similarity threshold to filter results
    similarity_threshold = 0.7
    relevant_results = [
        {"document": doc, "similarity": sim[0]}
        for doc, sim in zip(results["documents"], results["distances"])
        if sim[0] > similarity_threshold
    ]

    if relevant_results:
        return {"results": relevant_results}
    else:
        return {"message": "No relevant results found"}

# Run this FastAPI server with: uvicorn main:app --reload
# API docs are available at: http://127.0.0.1:8000/docs
