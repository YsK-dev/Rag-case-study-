from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
import sys
import os
import shutil
import logging
import time
import json
from datetime import datetime
from typing import Optional, List

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
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model mapping - frontend model names to Ollama models
MODEL_MAPPING = {
    "gemini-1.5-flash": "qwen3:1.7b",     # Fast model (smaller, quicker)
    "gemini-1.5-pro": "pielee/qwen3-4b-thinking-2507_q8:latest",  # Smart model (larger, better reasoning)
    "flash": "qwen3:1.7b",
    "pro": "pielee/qwen3-4b-thinking-2507_q8:latest"
}

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
    model: Optional[str] = "gemini-1.5-flash"
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7  # Temperature for LLM generation (0.0-1.0)

class FeedbackRequest(BaseModel):
    message_id: str
    feedback: str  # "up" or "down"

class ReasoningTrace(BaseModel):
    query_rewrite: str = ""
    search_terms: List[str] = []
    retrieval_strategy: str = ""

class SourceChunk(BaseModel):
    text: str
    source: str
    chunk_id: str = ""
    page: int = 0
    content_type: str = "text"
    file_path: str = ""
    char_count: int = 0
    token_count: int = 0
    char_count_raw: int = 0
    confidence: float = 0.5

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk] = []
    metadata: dict = {}
    reasoning: Optional[ReasoningTrace] = None

# Feedback storage path
FEEDBACK_FILE = "data/feedback.json"

def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return {"entries": []}

def save_feedback(data):
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# API Endpoints
@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse("static/index.html")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": llm_client.model}

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Store user feedback for a message"""
    logger.info(f"Received feedback: {request.feedback} for message {request.message_id}")
    
    try:
        feedback_data = load_feedback()
        feedback_data["entries"].append({
            "message_id": request.message_id,
            "feedback": request.feedback,
            "timestamp": datetime.now().isoformat()
        })
        save_feedback(feedback_data)
        
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_sse_stream(question: str, top_k: int, model_name: str, temperature: float = 0.7):
    """Generator for Server-Sent Events streaming response"""
    start_time = time.time()
    
    try:
        # Analyze query for reasoning trace
        reasoning = rag_engine.analyze_query(question)
        
        # Retrieve relevant context
        retrieval_start = time.time()
        context_chunks, metadatas = rag_engine.retrieve(question, top_k=top_k)
        retrieval_time = time.time() - retrieval_start
        
        logger.info(f"Retrieved {len(context_chunks)} chunks in {retrieval_time:.2f}s")
        
        if not context_chunks:
            # No context found, use LLM directly with streaming
            generation_start = time.time()
            llm_thinking = ""
            for result in llm_client.chat_stream(question):
                if result.get('token'):  # Filter empty tokens
                    yield f"data: {json.dumps({'token': result['token']})}\n\n"
                if result.get('thinking'):
                    llm_thinking += result['thinking']
                    yield f"data: {json.dumps({'thinking': result['thinking']})}\n\n"
            generation_time = time.time() - generation_start
            
            total_time = time.time() - start_time
            
            # Send empty sources
            yield f"data: {json.dumps({'sources': []})}\n\n"
            
            # Send metadata
            metadata = {
                "model": model_name,
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": round(generation_time, 3),
                "total_time": round(total_time, 3),
                "chunks_used": 0,
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps({'metadata': metadata})}\n\n"
            
            # Send reasoning (include LLM thinking if present)
            if llm_thinking:
                reasoning['llm_thinking'] = llm_thinking
            yield f"data: {json.dumps({'reasoning': reasoning})}\n\n"
            
            yield "data: [DONE]\n\n"
            return
        
        # Stream answer generation with context
        generation_start = time.time()
        llm_thinking = ""
        for result in llm_client.generate_answer_stream(question, context_chunks, temperature):
            if result.get('token'):  # Filter empty tokens
                yield f"data: {json.dumps({'token': result['token']})}\n\n"
            if result.get('thinking'):
                llm_thinking += result['thinking']
                yield f"data: {json.dumps({'thinking': result['thinking']})}\n\n"
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        # Format sources with metadata and confidence
        sources = []
        for i, (chunk, metadata) in enumerate(zip(context_chunks, metadatas)):
            source_file = metadata.get('source', 'Unknown')
            file_path = f"/api/documents/{source_file}" if source_file != 'Unknown' else ""
            
            sources.append({
                "text": chunk[:300] + "..." if len(chunk) > 300 else chunk,
                "source": source_file,
                "chunk_id": f"chunk_{i}",
                "page": metadata.get('page', 0),
                "content_type": metadata.get('type', 'text'),
                "file_path": file_path,
                "char_count": metadata.get('char_count', len(chunk)),
                "token_count": metadata.get('token_count', len(chunk) // 4),
                "confidence": metadata.get('confidence', 0.5)
            })
        
        # Send sources
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        
        # Send metadata
        response_metadata = {
            "model": model_name,
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3),
            "total_time": round(total_time, 3),
            "chunks_used": len(context_chunks),
            "timestamp": datetime.now().isoformat()
        }
        yield f"data: {json.dumps({'metadata': response_metadata})}\n\n"
        
        # Send reasoning (include LLM thinking if present)
        if llm_thinking:
            reasoning['llm_thinking'] = llm_thinking
        yield f"data: {json.dumps({'reasoning': reasoning})}\n\n"
        
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Answer questions using RAG with optional streaming
    """
    start_time = time.time()
    logger.info(f"Received question: {request.question}, stream={request.stream}, model={request.model}")
    
    # Get model name from mapping
    model_name = MODEL_MAPPING.get(request.model, "qwen3:1.7b")
    
    # Handle streaming request
    if request.stream:
        return StreamingResponse(
            generate_sse_stream(request.question, request.top_k, model_name, request.temperature),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    # Non-streaming response
    try:
        # Analyze query for reasoning trace
        reasoning = rag_engine.analyze_query(request.question)
        
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
                    "model": model_name,
                    "retrieval_time": round(retrieval_time, 3),
                    "generation_time": round(generation_time, 3),
                    "total_time": round(total_time, 3),
                    "chunks_used": 0,
                    "timestamp": datetime.now().isoformat()
                },
                reasoning=ReasoningTrace(**reasoning)
            )
        
        # Generate answer with context
        generation_start = time.time()
        answer = llm_client.generate_answer(request.question, context_chunks)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        logger.info(f"Answer generated in {generation_time:.2f}s (total: {total_time:.2f}s)")
        
        # Format sources with metadata and confidence
        sources = []
        for i, (chunk, metadata) in enumerate(zip(context_chunks, metadatas)):
            # Get file path for the uploaded document
            source_file = metadata.get('source', 'Unknown')
            file_path = f"/api/documents/{source_file}" if source_file != 'Unknown' else ""
            
            sources.append(SourceChunk(
                text=chunk[:300] + "..." if len(chunk) > 300 else chunk,
                source=source_file,
                chunk_id=f"chunk_{i}",
                page=metadata.get('page', 0),
                content_type=metadata.get('type', 'text'),
                file_path=file_path,
                char_count=metadata.get('char_count', len(chunk)),
                token_count=metadata.get('token_count', len(chunk) // 4),
                char_count_raw=metadata.get('char_count_raw', len(chunk)),
                confidence=metadata.get('confidence', 0.5)
            ))
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            metadata={
                "model": model_name,
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": round(generation_time, 3),
                "total_time": round(total_time, 3),
                "chunks_used": len(context_chunks),
                "timestamp": datetime.now().isoformat()
            },
            reasoning=ReasoningTrace(**reasoning)
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

@app.get("/api/documents/{filename}")
async def get_document(filename: str):
    """Serve documents for PDF preview"""
    # Check in uploads directory first
    file_path = Path("data/uploads") / filename
    
    if not file_path.exists():
        # Check in data directory
        file_path = Path("data") / filename
    
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
        file_size = 0
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            file_size = len(content)
            buffer.write(content)
        
        logger.info(f"File saved to: {file_path}")
        
        # Ingest the document
        ingest_start = time.time()
        result = rag_engine.ingest_file(file_path)
        ingest_time = time.time() - ingest_start
        
        total_time = time.time() - start_time
        logger.info(f"Document ingested in {ingest_time:.2f}s (total: {total_time:.2f}s)")
        
        # Parse chunks count from result message
        chunks_created = 0
        if "chunks" in result.lower():
            try:
                chunks_created = int(result.split()[2])  # "Successfully processed X chunks..."
            except:
                chunks_created = 0
        
        return {
            "status": "success",
            "message": result,
            "filename": file.filename,
            "file_size": file_size,
            "chunks_created": chunks_created,
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
