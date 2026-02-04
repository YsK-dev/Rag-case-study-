from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_engine import RAGEngine
from llm_client import OllamaClient

# Initialize FastAPI app
app = FastAPI(
    title="RAG Q&A API",
    description="Retrieval-Augmented Generation API for document-based question answering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components
print("Initializing RAG Engine...")
rag_engine = RAGEngine(db_path="./chroma_db_web", collection_name="web_docs")
print("Initializing Ollama LLM...")
llm_client = OllamaClient(model="qwen3:1.7b")
print("Server ready!")

# Pydantic models
class ChatRequest(BaseModel):
    question: str
    top_k: int = 3

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []

# API Endpoints
@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("static/index.html")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "qwen3:1.7b"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer questions using RAG
    """
    try:
        # Retrieve relevant context
        context_chunks = rag_engine.retrieve(request.question, top_k=request.top_k)
        
        if not context_chunks:
            # No context found, use LLM directly
            answer = llm_client.chat(request.question)
            return ChatResponse(
                answer=answer,
                sources=[]
            )
        
        # Generate answer with context
        answer = llm_client.generate_answer(request.question, context_chunks)
        
        return ChatResponse(
            answer=answer,
            sources=context_chunks
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF document
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest the document
        result = rag_engine.ingest_file(file_path)
        
        return {
            "status": "success",
            "message": result,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
