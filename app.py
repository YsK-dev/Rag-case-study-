import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add src to path BEFORE any relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, conint, confloat
import logging
import time
import json
from typing import Optional, List, Dict

# Detect test mode BEFORE heavy imports to avoid loading models
TESTING = os.getenv("RAG_TESTING") == "1"

if not TESTING:
    from rag_engine import RAGEngine
    from llm_client import OllamaClient, sanitize_input
else:
    # Import only sanitize_input for tests (lightweight)
    from llm_client import sanitize_input

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"

# Supported file types (aligned with RAGEngine)
SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".docx",
    ".xlsx", ".xlsm", ".csv",
    ".html", ".htm", ".pptx",
}

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
# Note: If using credentials (cookies/auth headers), replace "*" with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Cannot use True with allow_origins=["*"]
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
if TESTING:
    from testing_stubs import DummyRAGEngine, DummyLLMClient

    rag_engine_chroma = DummyRAGEngine()
    rag_engine_faiss = None
    rag_engines = {"chroma": rag_engine_chroma}
    rag_engine = rag_engine_chroma
    llm_client = DummyLLMClient()
else:
    print("Initializing RAG Engine (ChromaDB)...")
    rag_engine_chroma = RAGEngine(
        db_path=str(BASE_DIR / "chroma_db_web"),
        collection_name="web_docs",
        backend="chroma",
    )

    # Try to initialize FAISS (optional - graceful fallback if not installed)
    rag_engine_faiss = None
    try:
        print("Initializing RAG Engine (FAISS)...")
        rag_engine_faiss = RAGEngine(
            db_path=str(BASE_DIR / "faiss_db"),
            backend="faiss",
            embedding_model=rag_engine_chroma.embedding_model,
            reranker=rag_engine_chroma.reranker,
        )
        print("FAISS engine initialized successfully.")
    except Exception as e:
        logger.warning(f"FAISS not available, using ChromaDB only: {e}")
        print(f"FAISS not available ({e}), using ChromaDB only.")

    # Convenience mapping (only include FAISS if available)
    rag_engines = {"chroma": rag_engine_chroma}
    if rag_engine_faiss is not None:
        rag_engines["faiss"] = rag_engine_faiss

    # Keep a default alias for backward compatibility
    rag_engine = rag_engine_chroma

    print("Initializing Ollama LLM...")
    llm_client = OllamaClient(model="qwen3:1.7b")
    print("Server ready!")

# Pydantic models
class ChatRequest(BaseModel):
    question: str
    top_k: conint(ge=1, le=20) = 3
    model: Optional[str] = "gemini-1.5-flash"
    stream: Optional[bool] = False
    temperature: Optional[confloat(ge=0.0, le=1.0)] = 0.7  # Temperature for LLM generation (0.0-1.0)
    vector_store: Optional[str] = "chroma"  # "chroma" or "faiss"
    conversation_id: Optional[str] = None  # For multi-turn conversations
    source_filter: Optional[str] = None  # Filter by document name

class FeedbackRequest(BaseModel):
    message_id: str
    feedback: str  # "up" or "down"

class DeleteDocumentRequest(BaseModel):
    source: str  # filename to delete

class ReasoningTrace(BaseModel):
    query_rewrite: str = ""
    search_terms: List[str] = []
    retrieval_strategy: str = ""
    llm_thinking: Optional[str] = None
    source_filter: Optional[str] = None

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
    rerank_score: float = 0.0
    cosine_similarity: float = 0.0
    euclidean_distance: float = 0.0

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk] = []
    metadata: dict = {}
    reasoning: Optional[ReasoningTrace] = None

# Feedback storage path
FEEDBACK_FILE = DATA_DIR / "feedback.json"

def load_feedback():
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return {"entries": []}

def save_feedback(data):
    os.makedirs(FEEDBACK_FILE.parent, exist_ok=True)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# ── Conversation history store (in-memory, keyed by conversation_id) ──
# Each entry: list of {"role": "user"|"assistant", "content": str}
_conversations: Dict[str, List[Dict[str, str]]] = defaultdict(list)

MAX_HISTORY_MESSAGES = 20  # max messages (not turns) kept per conversation

def _get_history(conversation_id: Optional[str]) -> List[Dict[str, str]]:
    if not conversation_id:
        return []
    return _conversations[conversation_id][-MAX_HISTORY_MESSAGES:]

def _append_history(conversation_id: Optional[str], role: str, content: str):
    if not conversation_id:
        return
    _conversations[conversation_id].append({"role": role, "content": content})
    # Trim to avoid unbounded growth
    if len(_conversations[conversation_id]) > MAX_HISTORY_MESSAGES * 2:
        _conversations[conversation_id] = _conversations[conversation_id][-MAX_HISTORY_MESSAGES:]

# ── Chat history persistence (save messages per conversation to disk) ──
CHAT_HISTORY_DIR = DATA_DIR / "chat_history"

def _validate_conversation_id(conversation_id: str) -> bool:
    """Validate conversation_id to prevent path traversal attacks.
    
    Returns True if the ID is safe, False otherwise.
    """
    if not conversation_id:
        return False
    # Reject IDs with path separators, parent directory references, or hidden files
    if '/' in conversation_id or '\\' in conversation_id:
        return False
    if '..' in conversation_id:
        return False
    if conversation_id.startswith('.'):
        return False
    # Ensure the resolved path stays within CHAT_HISTORY_DIR
    try:
        target = (CHAT_HISTORY_DIR / f"{conversation_id}.json").resolve()
        chat_dir = CHAT_HISTORY_DIR.resolve()
        if not str(target).startswith(str(chat_dir)):
            return False
    except Exception:
        return False
    return True

def _save_chat_message(conversation_id: str, message: dict):
    """Append a chat message to the on-disk history for a conversation."""
    if not conversation_id or not _validate_conversation_id(conversation_id):
        return
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    history_file = CHAT_HISTORY_DIR / f"{conversation_id}.json"
    history: list = []
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(message)
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

def _load_chat_history(conversation_id: str) -> list:
    """Load persisted chat history for a conversation."""
    if not _validate_conversation_id(conversation_id):
        return []
    history_file = CHAT_HISTORY_DIR / f"{conversation_id}.json"
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

LOW_CONFIDENCE_MSG = (
    "I don't have enough information in the provided documents to answer "
    "this question confidently. Please try rephrasing or uploading a more "
    "relevant document."
)

# API Endpoints
@app.get("/")
async def root():
    """Serve the frontend if available, otherwise return API info."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "status": "ok",
        "message": "RAG API is running",
        "docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": llm_client.model}

@app.get("/api/vector-stores")
async def vector_stores():
    """Return available vector store backends and their stats."""
    return {
        "available": ["chroma", "faiss"],
        "default": "chroma",
        "stats": {
            name: engine.get_stats()
            for name, engine in rag_engines.items()
        },
    }

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

def generate_sse_stream(
    question: str,
    top_k: int,
    model_name: str,
    temperature: float = 0.7,
    vector_store: str = "chroma",
    conversation_id: Optional[str] = None,
    source_filter: Optional[str] = None,
):
    """Generator for Server-Sent Events streaming response"""
    start_time = time.time()
    
    # Select vector store engine
    engine = rag_engines.get(vector_store, rag_engine)

    # Conversation history for multi-turn
    chat_history = _get_history(conversation_id)
    
    try:
        # Analyze query for reasoning trace (may detect source filter)
        reasoning = engine.analyze_query(question)
        effective_filter = source_filter or reasoning.get("source_filter")
        
        # Retrieve relevant context
        retrieval_start = time.time()
        context_chunks, metadatas = engine.retrieve(
            question,
            top_k=top_k,
            source_filter=effective_filter,
        )
        retrieval_time = time.time() - retrieval_start
        
        logger.info(f"Retrieved {len(context_chunks)} chunks in {retrieval_time:.2f}s (backend={vector_store})")
        
        if not context_chunks:
            # No context found — return low-confidence message
            yield f"data: {json.dumps({'token': LOW_CONFIDENCE_MSG})}\n\n"
            generation_time = 0.0
            total_time = time.time() - start_time

            _append_history(conversation_id, "user", question)
            _append_history(conversation_id, "assistant", LOW_CONFIDENCE_MSG)
            _save_chat_message(conversation_id, {"role": "user", "content": question, "timestamp": datetime.now().isoformat()})
            _save_chat_message(conversation_id, {"role": "assistant", "content": LOW_CONFIDENCE_MSG, "timestamp": datetime.now().isoformat()})
            
            yield f"data: {json.dumps({'sources': []})}\n\n"
            metadata = {
                "model": model_name,
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": 0,
                "total_time": round(total_time, 3),
                "chunks_used": 0,
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps({'metadata': metadata})}\n\n"
            yield f"data: {json.dumps({'reasoning': reasoning})}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        # Stream answer generation with context + history
        generation_start = time.time()
        llm_thinking = ""
        full_answer = ""
        for result in llm_client.generate_answer_stream(question, context_chunks, temperature, chat_history, model=model_name):
            if result.get('token'):
                full_answer += result['token']
                yield f"data: {json.dumps({'token': result['token']})}\n\n"
            if result.get('thinking'):
                llm_thinking += result['thinking']
                yield f"data: {json.dumps({'thinking': result['thinking']})}\n\n"
        generation_time = time.time() - generation_start

        # Persist conversation turn
        _append_history(conversation_id, "user", question)
        _append_history(conversation_id, "assistant", full_answer)
        _save_chat_message(conversation_id, {"role": "user", "content": question, "timestamp": datetime.now().isoformat()})
        _save_chat_message(conversation_id, {"role": "assistant", "content": full_answer, "timestamp": datetime.now().isoformat()})
        
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
                "confidence": metadata.get('confidence', 0.5),
                "rerank_score": metadata.get('rerank_score', 0.0),
                "cosine_similarity": metadata.get('cosine_similarity', 0.0),
                "euclidean_distance": metadata.get('euclidean_distance', 0.0)
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
    Answer questions using RAG with optional streaming.
    Supports multi-turn conversations, guardrails, source filtering.
    """
    start_time = time.time()
    logger.info(f"Received question: {request.question}, stream={request.stream}, model={request.model}")

    # ── Guardrails: sanitise input ──
    cleaned_question, is_safe = sanitize_input(request.question)
    if not is_safe:
        logger.warning(f"Prompt injection detected: {request.question[:100]}")
        raise HTTPException(
            status_code=400,
            detail="Your message was flagged as a potential prompt injection and has been rejected."
        )
    cleaned_question = cleaned_question.strip()
    if not cleaned_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    # Get model name from mapping
    model_name = MODEL_MAPPING.get(request.model, "qwen3:1.7b")
    
    # Handle streaming request
    if request.stream:
        return StreamingResponse(
            generate_sse_stream(
                cleaned_question, request.top_k, model_name,
                request.temperature,
                request.vector_store or "chroma",
                request.conversation_id,
                request.source_filter,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    # Non-streaming response
    try:
        
        # Select vector store engine
        engine = rag_engines.get(request.vector_store or "chroma", rag_engine)
        
        # Analyze query for reasoning trace
        reasoning = engine.analyze_query(cleaned_question)
        effective_filter = request.source_filter or reasoning.get("source_filter")

        # Conversation history
        chat_history = _get_history(request.conversation_id)
        
        # Retrieve relevant context
        retrieval_start = time.time()
        context_chunks, metadatas = engine.retrieve(
            cleaned_question,
            top_k=request.top_k,
            source_filter=effective_filter,
        )
        retrieval_time = time.time() - retrieval_start
        
        logger.info(f"Retrieved {len(context_chunks)} chunks in {retrieval_time:.2f}s (backend={request.vector_store})")
        
        if not context_chunks:
            # No confident context — abstain
            logger.warning("No relevant context found above confidence threshold")
            total_time = time.time() - start_time

            _append_history(request.conversation_id, "user", cleaned_question)
            _append_history(request.conversation_id, "assistant", LOW_CONFIDENCE_MSG)
            _save_chat_message(
                request.conversation_id,
                {"role": "user", "content": cleaned_question, "timestamp": datetime.now().isoformat()},
            )
            _save_chat_message(
                request.conversation_id,
                {"role": "assistant", "content": LOW_CONFIDENCE_MSG, "timestamp": datetime.now().isoformat()},
            )

            return ChatResponse(
                answer=LOW_CONFIDENCE_MSG,
                sources=[],
                metadata={
                    "model": model_name,
                    "retrieval_time": round(retrieval_time, 3),
                    "generation_time": 0,
                    "total_time": round(total_time, 3),
                    "chunks_used": 0,
                    "timestamp": datetime.now().isoformat()
                },
                reasoning=ReasoningTrace(**reasoning)
            )
        
        # Generate answer with context + history
        generation_start = time.time()
        answer, _thinking = llm_client.generate_answer(
            cleaned_question, context_chunks, request.temperature, chat_history, model=model_name
        )
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        logger.info(f"Answer generated in {generation_time:.2f}s (total: {total_time:.2f}s)")

        # Persist history (both in-memory and on-disk)
        _append_history(request.conversation_id, "user", cleaned_question)
        _append_history(request.conversation_id, "assistant", answer)
        _save_chat_message(request.conversation_id, {"role": "user", "content": cleaned_question, "timestamp": datetime.now().isoformat()})
        _save_chat_message(request.conversation_id, {"role": "assistant", "content": answer, "timestamp": datetime.now().isoformat()})
        
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
                confidence=metadata.get('confidence', 0.5),
                rerank_score=metadata.get('rerank_score', 0.0),
                cosine_similarity=metadata.get('cosine_similarity', 0.0),
                euclidean_distance=metadata.get('euclidean_distance', 0.0)
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
            reasoning=ReasoningTrace(**reasoning, llm_thinking=_thinking)
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pdf/{filename}")
async def serve_pdf(filename: str):
    """Legacy PDF endpoint — redirects to the generic documents endpoint."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/api/documents/{filename}")

# ── Document management endpoints ──

@app.get("/api/documents")
async def list_documents(vector_store: str = "chroma"):
    """List all documents stored in the vector store."""
    if vector_store not in rag_engines:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid vector_store '{vector_store}'. Use one of: {', '.join(rag_engines.keys())}."
        )
    engine = rag_engines[vector_store]
    docs = engine.list_documents()
    return {"documents": docs, "vector_store": vector_store}

@app.delete("/api/documents/{filename}")
async def delete_document_endpoint(filename: str, vector_store: str = "chroma"):
    """Delete a document and all its chunks from the vector store."""
    safe_name = Path(filename).name
    engine = rag_engines.get(vector_store, rag_engine)
    result = engine.delete_document(safe_name)

    # Also delete from the other backend
    other = "faiss" if vector_store == "chroma" else "chroma"
    other_engine = rag_engines.get(other)
    if other_engine:
        other_engine.delete_document(safe_name)

    # Optionally remove file from uploads
    file_path = UPLOAD_DIR / safe_name
    if file_path.exists():
        os.remove(file_path)

    return {"status": "success", "message": result}

# ── Chat history persistence endpoints ──

@app.get("/api/chat/history/{conversation_id}")
async def get_chat_history(conversation_id: str):
    """Return persisted chat history for a conversation."""
    history = _load_chat_history(conversation_id)
    return {"conversation_id": conversation_id, "messages": history}

@app.delete("/api/chat/history/{conversation_id}")
async def clear_chat_history(conversation_id: str):
    """Clear persisted chat history for a conversation."""
    if not _validate_conversation_id(conversation_id):
        raise HTTPException(status_code=400, detail="Invalid conversation_id")
    history_file = CHAT_HISTORY_DIR / f"{conversation_id}.json"
    if history_file.exists():
        os.remove(history_file)
    if conversation_id in _conversations:
        del _conversations[conversation_id]
    return {"status": "success", "message": f"History for {conversation_id} cleared"}

@app.get("/api/chat/conversations")
async def list_conversations():
    """List all persisted conversation IDs."""
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    files = sorted(CHAT_HISTORY_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    convos = []
    for f in files:
        cid = f.stem
        try:
            with open(f, "r") as fh:
                msgs = json.load(fh)
            convos.append({
                "conversation_id": cid,
                "message_count": len(msgs),
                "last_message": msgs[-1].get("content", "")[:80] if msgs else "",
            })
        except Exception:
            convos.append({"conversation_id": cid, "message_count": 0, "last_message": ""})
    return {"conversations": convos}

@app.get("/api/documents/{filename}")
async def get_document(filename: str):
    """Serve uploaded documents for preview (PDF, DOCX, XLSX, CSV, MD, TXT, HTML, PPTX)."""
    safe_name = Path(filename).name
    ext = Path(safe_name).suffix.lower()

    mime_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }

    if ext not in mime_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}"
        )

    file_path = UPLOAD_DIR / safe_name
    if not file_path.exists():
        file_path = DATA_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    return FileResponse(
        file_path,
        media_type=mime_map[ext],
        filename=safe_name
    )

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document (PDF, DOCX, XLSX, CSV, TXT, MD, HTML, PPTX)
    """
    start_time = time.time()
    logger.info(f"Uploading file: {file.filename}")
    
    try:
        # Validate file type
        safe_name = Path(file.filename).name
        file_ext = Path(safe_name).suffix.lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            logger.warning(f"Invalid file type: {file.filename}")
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {allowed}"
            )
        
        # Save uploaded file
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        file_path = UPLOAD_DIR / safe_name
        
        # Delete old chunks if re-uploading (prevents duplicates)
        rag_engine_chroma.delete_document(safe_name)
        if rag_engine_faiss is not None:
            rag_engine_faiss.delete_document(safe_name)
        
        file_size = 0
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            file_size = len(content)
            buffer.write(content)
        
        logger.info(f"File saved to: {file_path}")
        
        # Ingest the document into available backends
        ingest_start = time.time()
        result_chroma = rag_engine_chroma.ingest_file(file_path)
        if rag_engine_faiss is not None:
            rag_engine_faiss.ingest_file(file_path)
        ingest_time = time.time() - ingest_start
        
        # Use the chroma result as the primary message
        result = result_chroma
        
        total_time = time.time() - start_time
        logger.info(f"Document ingested in {ingest_time:.2f}s (total: {total_time:.2f}s)")
        
        # Parse chunks count from result message
        chunks_created = 0
        if "chunks" in result.lower():
            try:
                chunks_created = int(result.split()[2])  # "Successfully processed X chunks..."
            except (ValueError, IndexError):
                chunks_created = 0
        
        return {
            "status": "success",
            "message": result,
            "filename": safe_name,
            "file_size": file_size,
            "chunks_created": chunks_created,
            "processing_time": round(total_time, 2)
        }
        
    except HTTPException:
        # Re-raise expected HTTP errors (e.g., invalid file type)
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{filename}/preview")
async def preview_document(filename: str):
    """Extract text/table content from non-PDF documents for in-browser preview."""
    safe_name = Path(filename).name
    ext = Path(safe_name).suffix.lower()

    previewable = {".docx", ".xlsx", ".xlsm", ".csv", ".pptx", ".txt", ".md", ".html", ".htm"}
    if ext not in previewable:
        raise HTTPException(status_code=400, detail=f"Preview not supported for {ext}")

    file_path = UPLOAD_DIR / safe_name
    if not file_path.exists():
        file_path = DATA_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        if ext in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                return {"type": "text", "content": fh.read(), "filename": safe_name}

        if ext in (".html", ".htm"):
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                return {"type": "html", "content": fh.read(), "filename": safe_name}

        if ext == ".csv":
            import csv as csv_mod
            rows = []
            with open(file_path, "r", newline="", encoding="utf-8", errors="replace") as fh:
                reader = csv_mod.reader(fh)
                for row in reader:
                    rows.append(row)
            return {"type": "table", "sheets": [{"name": safe_name, "rows": rows}], "filename": safe_name}

        if ext in (".xlsx", ".xlsm"):
            from openpyxl import load_workbook as _load_wb
            wb = _load_wb(str(file_path), data_only=True, read_only=True)
            sheets = []
            for sheet in wb.worksheets:
                rows = []
                for row in sheet.iter_rows(values_only=True):
                    rows.append(["" if c is None else str(c) for c in row])
                sheets.append({"name": sheet.title, "rows": rows})
            return {"type": "table", "sheets": sheets, "filename": safe_name}

        if ext == ".docx":
            from docx import Document as _Document
            doc = _Document(str(file_path))
            paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
            tables = []
            for table in doc.tables:
                tbl_rows = []
                for row in table.rows:
                    tbl_rows.append([cell.text.strip() for cell in row.cells])
                tables.append(tbl_rows)
            return {
                "type": "docx",
                "paragraphs": paragraphs,
                "tables": tables,
                "filename": safe_name,
            }

        if ext == ".pptx":
            from pptx import Presentation as _Presentation
            prs = _Presentation(str(file_path))
            slides = []
            for slide in prs.slides:
                texts = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        texts.append(shape.text)
                slides.append("\n".join(texts))
            return {"type": "pptx", "slides": slides, "filename": safe_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview extraction failed: {str(e)}") from e

    raise HTTPException(status_code=400, detail=f"Preview not supported for {ext}")

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    logger.warning("Static directory not found; skipping static file mount.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
