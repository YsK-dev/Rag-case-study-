# RAG Case Study - Document Q&A System

A Retrieval-Augmented Generation (RAG) system that combines document retrieval with local LLM inference using Ollama.

## 🎯 Overview

This project demonstrates a complete RAG pipeline:
- **Document Ingestion**: PDF processing and chunking
- **Vector Embeddings**: Semantic text representation using sentence-transformers
- **Vector Database**: ChromaDB for efficient similarity search
- **LLM Integration**: Ollama (qwen3:1.7b) for answer generation
- **Web Interface**: Modern FastAPI backend + responsive frontend

### Features

- 📤 **Upload PDFs**: Drag and drop or click to upload documents
- 💬 **Chat Interface**: Ask questions about your documents
- 🎨 **Modern UI**: Dark mode with glassmorphism and smooth animations
- ⚡ **Real-time**: Instant responses with loading indicators
- 📱 **Responsive**: Works on desktop, tablet, and mobile

## 🖥️ CLI Demo

For a command-line interface:

```bash
python demo.py
```

## 📖 Usage

### Web Interface

1. **Upload a Document**: Click "Choose PDF File" and select a PDF
2. **Ask Questions**: Type your question in the chat input
3. **Get Answers**: The AI will retrieve relevant context and generate answers
4. **Try Examples**: Click example questions to get started quickly

### Example Questions

- "What is RAG?"
- "What are the key components of a RAG system?"
- "What are the benefits of using RAG?"
- "What are common use cases for RAG?"


**Request:**
```json
{
  "question": "What is RAG?",
  "top_k": 3
}
```

**Response:**
```json
{
  "answer": "RAG (Retrieval-Augmented Generation) is...",
  "sources": ["chunk1", "chunk2", "chunk3"]
}
```

### POST `/api/upload`
Upload and ingest PDF documents

**Request:** `multipart/form-data` with PDF file

## 📝 Adding Your Own Documents

### Via Web Interface
Simply click "Choose PDF File" and upload your document

### Via Python
```python
from src.rag_engine import RAGEngine

rag = RAGEngine()
rag.ingest_file("path/to/your/document.pdf")
```


- **Model**: Change `model="qwen3:1.7b"` to any Ollama model
- **Port**: Modify `uvicorn.run(app, host="0.0.0.0", port=8000)`
- **Chunk Size**: Modify `chunk_size` in `src/rag_engine.py`
- **Top-K Results**: Adjust `top_k` parameter in chat requests

