from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
import os
import shutil
import logging
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_engine import RAGEngine
from llm_client import OllamaClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

class SourceChunk(BaseModel):
    text: str
    source: str
    chunk_id: str
    page: int = 0
    content_type: str = "text"
    file_path: str = ""
    char_count: int = 0
    token_count: int = 0
    char_count_raw: int = 0

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = []
    metadata: dict = {}

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
    start_time = time.time()
    logger.info(f"Received question: {request.question}")
    
    try:
        # Retrieve relevant context
        retrieval_start = time.time()
        context_chunks, metadatas = rag_engine.retrieve(request.question, top_k=request.top_k)
        retrieval_time = time.time() - retrieval_start
        
        logger.info(f"Retrieved {len(context_chunks)} chunks in {retrieval_time:.2f}s")
        
        if not context_chunks:
            # No context found, use LLM directly
            logger.warning("No relevant context found, using LLM without RAG")
            generation_start = time.time()
            answer = llm_client.chat(request.question)
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            logger.info(f"Response generated in {generation_time:.2f}s (total: {total_time:.2f}s)")
            
            return ChatResponse(
                answer=answer,
                sources=[],
                metadata={
                    "model": "qwen3:1.7b",
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": total_time,
                    "chunks_used": 0,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Generate answer with context
        generation_start = time.time()
        answer = llm_client.generate_answer(request.question, context_chunks)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        logger.info(f"Answer generated in {generation_time:.2f}s (total: {total_time:.2f}s)")
        
        # Format sources with metadata
        sources = []
        for i, (chunk, metadata) in enumerate(zip(context_chunks, metadatas)):
            # Get file path for the uploaded document
            source_file = metadata.get('source', 'Unknown')
            file_path = f"/api/pdf/{source_file}" if source_file != 'Unknown' else ""
            
            sources.append(SourceChunk(
                text=chunk[:200] + "..." if len(chunk) > 200 else chunk,
                source=source_file,
                chunk_id=f"chunk_{i}",
                page=metadata.get('page', 0),
                content_type=metadata.get('type', 'text'),
                file_path=file_path,
                char_count=metadata.get('char_count', 0),
                token_count=metadata.get('token_count', 0),
                char_count_raw=metadata.get('char_count_raw', 0)
            ))
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            metadata={
                "model": "qwen3:1.7b",
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "chunks_used": len(context_chunks),
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pdf/{filename}")
async def serve_pdf(filename: str):
    """
    Serve PDF files for viewing
    """
    # Check in uploads directory
    file_path = os.path.join("data/uploads", filename)
    
    if not os.path.exists(file_path):
        # Check in data directory
        file_path = os.path.join("data", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return FileResponse(file_path, media_type="application/pdf")

#for review Pdfs 
@app.get("/api/documents/{filename}")
async def get_document(filename: str):
    file_path = Path(f"./uploads/{filename}")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename
    )

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF document
    """
    start_time = time.time()
    logger.info(f"Uploading file: {file.filename}")
    
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            logger.warning(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved to: {file_path}")
        
        # Ingest the document
        ingest_start = time.time()
        result = rag_engine.ingest_file(file_path)
        ingest_time = time.time() - ingest_start
        
        total_time = time.time() - start_time
        logger.info(f"Document ingested in {ingest_time:.2f}s (total: {total_time:.2f}s)")
        
        return {
            "status": "success",
            "message": result,
            "filename": file.filename,
            "processing_time": round(total_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
