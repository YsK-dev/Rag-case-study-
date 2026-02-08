
# 6. Literature Review & Comparative Analysis

## Resources Used

### Official Documentation

| Resource                | Usage                                              |
| ----------------------- | -------------------------------------------------- |
| [LangChain Documentation](https://docs.langchain.com/langsmith/evaluate-rag-tutorial#evaluate-a-rag-application) | RAG architectures, retrievers, chunking strategies |
| [ChromaDB Documentation](https://docs.trychroma.com/docs/overview/getting-started)  | Local vector storage, similarity search            |
| [Ollama Documentation](https://docs.ollama.com/)    | Local LLM serving and model lifecycle              |
| [FastAPI Documentation](https://fastapi.tiangolo.com/tutorial/)   | Async API design, streaming responses              |
| [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)              | High-performance similarity search and indexing    |

### Research Papers & Technical Articles

| Source                                                                                  | Key Contributions                     |
| --------------------------------------------------------------------------------------- | ------------------------------------- |
| Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | Foundational RAG architecture         |
| Robertson et al., *BM25 and Beyond*                                                     | Keyword-based relevance scoring       |
| Pinecone Blog – Chunking Strategies                                                     | Empirical chunk size trade-offs       |
| OpenAI Cookbook – Embeddings for QA                                                     | Practical embedding workflows         |
| SBERT Documentation                                                                     | Semantic search model benchmarks      |
| MTEB Leaderboard (Hugging Face)                                                         | Embedding model comparison            |
| EvidentlyAI – RAG Use Cases                                                             | Real-world application patterns       |
| Confident AI – RAG Evaluation Metrics                                                   | Faithfulness, relevancy, grounding    |
| Hugging Face Cookbook – LLM as Judge                                                    | Automated RAG evaluation              |
| Py-PDF Benchmarks                                                                       | PDF extraction performance comparison |

---

## Semantic Search as a UX Primitive

A critical insight from the literature is that **semantic search is not merely a retrieval technique, but a UX feature**. Users rarely express their intent explicitly or with correct terminology. [Semantic embeddings](https://sbert.net/docs/sentence_transformer/pretrained_models.html#semantic-search-models)  allow the system to infer intent even when the query is vague, incomplete, or phrased differently from the source documents.

Sentence-Transformers (SBERT) models are explicitly designed for this purpose and outperform traditional keyword-based approaches in semantic similarity tasks, as shown in the SBERT benchmark suite. This aligns with the core RAG principle: *retrieval quality directly determines generation quality*.

> **Golden Rule**: *Garbage in → Garbage out*.
> If retrieval fails, even the strongest LLM will hallucinate.

---

## Comparative Analysis

### Vector Database Selection

| Database     | Pros                                      | Cons                      | Decision   |
| ------------ | ----------------------------------------- | ------------------------- | ---------- |
| **ChromaDB** | Zero-config, SQLite-backed, Python-native | Single-node only          | ✅ Primary  |
| **FAISS**    | GPU support, billion-scale vectors        | No persistence by default | ✅ Optional |
| Pinecone     | Fully managed, scalable                   | Cloud-only, cost          | ❌ Privacy  |
| Weaviate     | Hybrid search, GraphQL                    | Heavy deployment          | ❌ Overhead |
| Milvus       | Distributed, GPU-native                   | Operational complexity    | ❌ Overkill |

**Rationale**
ChromaDB provides the best developer experience for **local-first RAG systems**, while FAISS remains an optional performance backend for advanced use cases.

---

### LLM Inference Options

| Option                    | Pros                            | Cons                      | Decision      |
| ------------------------- | ------------------------------- | ------------------------- | ------------- |
| **Ollama**                | Simple CLI, REST API, model hub | Limited advanced controls | ✅ Primary     |
| llama.cpp                 | Maximum performance             | Manual setup              | ❌ DX          |
| vLLM                      | Production-grade batching       | GPU-only                  | ❌ Hardware    |
| Hugging Face Transformers | Large ecosystem                 | Slow on CPU               | ❌ Performance |
| OpenAI API                | Best raw quality                | Cloud, cost, privacy      | ❌ Privacy     |

**Rationale**
Ollama abstracts away quantization, model management, and serving, enabling fast iteration without sacrificing locality.

---

### Embedding Models

| Model                   | Dim  | Speed  | Quality   | Decision    |
| ----------------------- | ---- | ------ | --------- | ----------- |
| **all-MiniLM-L6-v2**    | 384  | Fast   | Good      | ✅ Default   |
| all-mpnet-base-v2       | 768  | Medium | Better    | Alternative |
| BGE-large               | 1024 | Slow   | Excellent | ❌ Memory    |
| OpenAI text-embedding-3 | 1536 | API    | Excellent | ❌ Cloud     |

**Rationale**
MiniLM provides an excellent speed-to-quality ratio and consistently ranks well on MTEB semantic retrieval tasks while remaining CPU-friendly.
[see here: mteb leadorboard](https://huggingface.co/spaces/mteb/leaderboard)

---

### Retrieval Strategies

| Strategy        | Use Case            | Status    |
| --------------- | ------------------- | --------- |
| Semantic Search | Conceptual queries  | ✅         |
| BM25            | Exact term matching | ✅         |
| Hybrid (RRF)    | General queries     | ✅ Default |
| Reranking       | High precision      | ✅         |
| HyDE            | Query expansion     | ❌ Future  |

**Rationale**
Hybrid retrieval using Reciprocal Rank Fusion (RRF) consistently outperforms single-method retrieval by combining semantic understanding with lexical precision.

---

## Document Parsing: PDFPlumber vs PyMuPDF

[Benchmark](https://github.com/py-pdf/benchmarks) results indicate that **PyMuPDF (fitz)** significantly outperforms PDFPlumber in speed and memory usage, particularly for large or complex PDFs. Consequently, PyMuPDF was selected as the default parser, with PDFPlumber retained as a fallback for edge cases involving layout-sensitive extraction.

---

## Evaluation & Testing Strategy

RAG evaluation requires more than traditional accuracy metrics. Based on recent research and industry practice, the following dimensions were prioritized:

* **Answer Relevancy**
* **Faithfulness / Grounding**
* **Context Utilization**
* **Hallucination Rate**

To automate evaluation, **[LLMs-as-a-Judge](https://huggingface.co/learn/cookbook/en/llm_judge)** were employed using Groq’s free tier, following Hugging Face’s evaluation cookbook. This approach enables scalable, repeatable qualitative assessment without human labeling.
[Eval metrics](https://www.confident-ai.com/blog/rag-evaluation-metrics-answer-relevancy-faithfulness-and-more)

---

## Model Selection Constraints

Due to hardware limitations, large frontier models were impractical. But if we can we  would use , **Qwen3-30B-A3B-Instruct-2507** was also used for annotation tasks, as demonstrated in the Hugging Face :
[TurkWeb-Edu-AnnotationsV3](https://huggingface.co/datasets/YsK-dev/TurkWeb-Edu-AnnotationsV3) dataset so we know it is good model to keep in mimd. For future iterations, **Gemma-3n-E2B-IT** is a promising candidate due to its on-device optimization and multimodal capabilities, enabling potential expansion into image and voice-based RAG.

---

## Lessons Learned

1. **Chunking is critical**: 500 tokens with 100 overlap balances recall and coherence.
2. **Semantic search defines UX**: Users don’t think in keywords.
3. **Hybrid retrieval wins**: Pure semantic or lexical search is insufficient alone.
4. **Caching matters**: Rebuilding BM25 indexes is prohibitively expensive.
5. **Evaluation is mandatory**: LLM judges catch failures humans miss.
6. **Local-first is viable**: With modern tooling, cloud APIs are optional.

---

