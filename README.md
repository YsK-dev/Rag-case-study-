# RAG Case Study - Local Document Q&A System

A **privacy-first, fully local** Retrieval-Augmented Generation (RAG) system that enables intelligent Q&A over your documents using Ollama and ChromaDB.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Next.js](https://img.shields.io/badge/Next.js-16+-black)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What This Project Does

Transform your document collection into an interactive knowledge base:

![alt text](image-3.png)

**Key Features:**
- 🔒 **100% Local**: No data leaves your machine
- 📄 **Multi-format**: PDF, DOCX, TXT, XLSX, CSV, PPTX, HTML, Markdown
- 🔍 **Hybrid Search**: Semantic + BM25 for best results
- 💬 **Streaming Chat**: Real-time responses with source citations
- 🎨 **Modern UI**: React/Next.js with dark mode
- 🧪 **Evaluated**: LLM-as-Judge scoring (avg 7.2/10)

---

## 🚀 Installation & Setup

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Required |
| Node.js | 18+ | For frontend |
| Ollama | Latest | [Download here](https://ollama.ai/) |

### Step 1: Clone the Repository

```bash
git clone https://github.com/YsK-dev/Rag-case-study-.git 
cd Rag-case-study
```

### Step 2: Install Ollama Models

```bash
# Install Ollama (macOS)
brew install ollama
# On Winows
winget install Ollama.Ollama
# Linux (Ubuntu / Debian / Arch / Fedora etc.)
curl -fsSL https://ollama.com/install.sh | sh


# Start Ollama service
ollama serve

# Pull required models (in another terminal)
ollama pull qwen3:1.7b                           # Fast model (⚡ 1.7B params)
ollama pull pielee/qwen3-4b-thinking-2507_q8     # Smart model (🧠 4B params, reasoning)
```

**Available Models:**

| Model ID | Label | Tier | Parameters | Best For |
|----------|-------|------|------------|----------|
| `qwen3-1.7b` | Qwen3 1.7B | ⚡ Fast | 1.7B | Quick responses, low latency |
| `qwen3-4b-thinking` | Qwen3 4B Thinking | 🧠 Smart | 4B | Complex reasoning, chain-of-thought |

### Step 3: Setup Backend (Python)

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

**Backend Dependencies (`requirements.txt`):**
```
fastapi          # Web framework
uvicorn          # ASGI server
chromadb         # Vector database
sentence-transformers  # Embedding model
pypdf            # PDF parsing
pymupdf          # PDF rendering
python-docx      # DOCX support
openpyxl         # Excel support
python-pptx      # PowerPoint support
beautifulsoup4   # HTML parsing
rank-bm25        # BM25 scoring
ollama           # LLM client
pytest           # Testing
```

### Step 4: Setup Frontend (Next.js)

```bash
cd frontedRag/rag-app

# Install Node.js dependencies
npm install
```

**Frontend Dependencies (`package.json`):**
```
next             # React framework (v16)
react            # UI library (v19)
tailwindcss      # Styling (v4)
framer-motion    # Animations
lucide-react     # Icons
react-pdf        # PDF preview
```

---

## 🏃 Running the Application

### Terminal 1: Start Ollama
```bash
ollama serve
# Runs on http://localhost:11434
```

### Terminal 2: Start Backend
```bash
uvicorn main:app --reload

# Runs on http://localhost:8000
```

### Terminal 3: Start Frontend
```bash
cd frontedRag/rag-app
npm run dev
# Runs on http://localhost:3000
```

### Verify Everything Works
```bash
# Check backend health
curl http://localhost:8000/api/health

# Check available models
curl http://localhost:8000/api/models
```

---

## 📖 Usage

### Web Interface
1. Open `http://localhost:3000` in your browser
2. **Upload documents** via drag-and-drop
3. **Select model** (Fast ⚡ or Smart 🧠)
4. **Ask questions** in natural language
5. **View answers** with source citations

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Ask questions (supports streaming) |
| `/api/upload` | POST | Upload documents |
| `/api/documents` | GET | List indexed documents |
| `/api/documents/{file}` | DELETE | Remove document |
| `/api/models` | GET | List available LLM models |
| `/api/health` | GET | Health check |

### Example API Request
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG?",
    "model": "qwen3-1.7b",
    "stream": true,
    "top_k": 5
  }'
```

---

## 🏗️ Project Structure

```
Rag-case-study/
├── app.py                 # FastAPI server (API layer)
├── requirements.txt       # Python dependencies
├── src/
│   ├── rag_engine.py      # Vector DB + retrieval logic
│   └── llm_client.py      # Ollama integration + guardrails
├── frontedRag/rag-app/    # Next.js frontend
│   ├── package.json       # Node.js dependencies
│   ├── app/               # Next.js app router
│   └── components/        # React components
├── data/
│   ├── uploads/           # Uploaded documents
│   └── eval_cases.json    # Evaluation test cases
├── tests/
│   ├── test_app.py        # Unit tests (24 tests)
│   └── eval_judge.py      # LLM-as-Judge evaluation
├── chroma_db/             # ChromaDB vector store
└── docs/                  # Detailed documentation
```

**Tech Stack:**
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Vector Store**: ChromaDB (default), FAISS (optional)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Ollama with qwen3 models
- **Frontend**: Next.js 16, React 19, TailwindCSS 4

---

## 🧪 Testing & Evaluation

### Unit Tests
```bash
RAG_TESTING=1 pytest tests/test_app.py -v
# 24 tests covering all endpoints
```

### LLM-as-Judge Evaluation
```bash
# Set Groq API key (get free at console.groq.com)
export GROQ_API_KEY=gsk_...

# Run evaluation
python tests/eval_judge.py --input data/eval_cases.json --output data/eval_results.jsonl
```

**Latest Evaluation Results:**
| Metric | Score |
|--------|-------|
| Overall | 7.19/10 |
| qwen3-32b judge | 7.98/10 |
| gpt-oss-120b judge | 7.14/10 |

---

## ⚙️ Configuration

| Setting | Location | Default |
|---------|----------|---------|
| LLM Models | `app.py` → `MODEL_REGISTRY` | qwen3-1.7b, qwen3-4b-thinking |
| Embedding Model | `rag_engine.py` | all-MiniLM-L6-v2 |
| Chunk Size | `rag_engine.py` | 500 tokens |
| Chunk Overlap | `rag_engine.py` | 100 tokens |
| Backend Port | `app.py` | 8000 |
| Frontend Port | `next.config.js` | 3000 |

### Adding New Ollama Models

1. Pull the model:
   ```bash
   ollama pull <model-name>
   ```

2. Add to `MODEL_REGISTRY` in `app.py`:
   ```python
   MODEL_REGISTRY = {
       "new-model": {
           "ollama_name": "<model-name>",
           "label": "Display Name",
           "tier": "fast",  # or "smart"
           "description": "Model description",
           "params": "7B",
       },
   }
   ```

---

## 📚 Documentation

Detailed documentation is available in [`docs/`](docs/):

| Document | Topic |
|----------|-------|
| [01_problem_definition.md](docs/01_problem_definition.md) | User needs & solution overview |
| [02_llm_layer.md](docs/02_llm_layer.md) | LLM integration & guardrails |
| [03_vector_layer.md](docs/03_vector_layer.md) | Chunking, embeddings, hybrid search |
| [04_api_layer.md](docs/04_api_layer.md) | API design & endpoints |
| [05_architecture.md](docs/05_architecture.md) | System architecture diagrams |
| [06_literature_review.md](docs/06_literature_review.md) | Technology comparisons |

---

## 📈 Development Timeline

| Date | Milestone |
|------|-----------|
| Feb 4 | Initial RAG engine + CLI demo |
| Feb 5 | PDF preview endpoint |
| Feb 6 | Frontend: React + Next.js |
| Feb 7 | Inline citations, reranker, bug fixes |
| Feb 8 | Unit tests + LLM-as-Judge evaluation |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ollama: command not found` | Install Ollama: `brew install ollama` |
| `Connection refused :11434` | Start Ollama: `ollama serve` |
| `Model not found` | Pull model: `ollama pull qwen3:1.7b` |
| `CUDA out of memory` | Use smaller model or CPU mode |
| Frontend build fails | Ensure Node.js 18+: `node --version` |

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `RAG_TESTING=1 pytest tests/`
5. Submit a pull request
