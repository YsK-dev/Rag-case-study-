import os
import chromadb
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, CrossEncoder
import pdfplumber
#import pymupdf as fitz
import re

class RAGEngine:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "docs_collection"):
        """
        Initializes the Enhanced RAG Engine with better PDF processing.
        """
        # 1. Initialize Vector DB (ChromaDB)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # 2. Create or get a collection
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        # 3. Initialize Embedding Model | we use all-MiniLM-L6-v2 but we could try even more like all-MiniLM-L12-v2  all-mpnet-base-v2 or if we deal with türkçe we could use newmindai/TurkEmbed4Retrieval boun-tabilab/TabiBERT/dbmdz/bert-base-turkish-cased
        print("Loading embedding model... (this happens only once)")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 4. Initialize Reranker (Method learned from LightRAG)
        # CrossEncoder is slower but much more accurate for the final scoring
        print("Loading reranker model...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("Models loaded.")

    def _detect_content_type(self, text: str) -> str:
        """Detect if text contains equations, code, or tables."""
        # Check for LaTeX/math symbols (prioritize this check)
        math_indicators = [
            r'\$\$.*?\$\$',  # Display math
            r'\$.*?\$',      # Inline math
            r'\\text\{',     # LaTeX text command
            r'\\frac\{',     # Fractions
            r'\\sum',        # Summation
            r'\\int',        # Integral
            r'\\left',       # Left delimiter
            r'\\right',      # Right delimiter
            r'[∑∫∂∇≈≠≤≥±×÷]', # Math symbols
        ]
        
        # Count math indicators
        math_count = sum(1 for pattern in math_indicators if re.search(pattern, text))
        
        # If we have multiple math indicators, it's likely an equation
        if math_count >= 2:
            return "equation"
        
        # Check for code (only if not math)
        code_patterns = [
            r'^\s{4,}\w+',  # Indented code
            r'\bdef\s+\w+\s*\(',  # Python function
            r'\bclass\s+\w+',     # Class definition
            r'\bimport\s+\w+',    # Import statement
            r'\breturn\s+',       # Return statement
            r'^\w+\s*=\s*\w+\(',  # Variable assignment with function call
        ]
        
        code_count = sum(1 for pattern in code_patterns if re.search(pattern, text, re.MULTILINE))
        if code_count >= 2:
            return "code"
        
        # Check for table-like structure
        if re.search(r'\|.*\|.*\|', text) or text.count('\t') > 3:
            return "table"
        
        return "text"

    def _extract_with_layout(self, file_path: str) -> List[Dict]:
        """
        Extract text from PDF with layout preservation.
        Returns list of chunks with metadata.
        """
        chunks = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text with layout preservation
                text = page.extract_text(layout=True)
                
                if not text or not text.strip():
                    continue
                
                # Extract tables separately
                tables = page.extract_tables()
                
                # Split page into paragraphs (double newline)
                paragraphs = re.split(r'\n\n+', text)
                
                for para in paragraphs:
                    para = para.strip()
                    if len(para) < 50:  # Skip very short paragraphs
                        continue
                    
                    content_type = self._detect_content_type(para)
                    
                    chunks.append({
                        'text': para,
                        'page': page_num,
                        'type': content_type,
                        'source': os.path.basename(file_path)
                    })
                
                # Add tables as separate chunks
                for table_idx, table in enumerate(tables):
                    if table:
                        # Format table as markdown
                        table_text = self._format_table(table)
                        chunks.append({
                            'text': f"[TABLE]\n{table_text}",
                            'page': page_num,
                            'type': 'table',
                            'source': os.path.basename(file_path)
                        })
        
        return chunks

    def _format_table(self, table: List[List]) -> str:
        """Format table data as markdown."""
        if not table:
            return ""
        
        lines = []
        for row in table:
            # Clean None values
            clean_row = [str(cell) if cell else "" for cell in row]
            lines.append("| " + " | ".join(clean_row) + " |")
        
        # Add header separator after first row
        if len(lines) > 1:
            num_cols = len(table[0])
            separator = "| " + " | ".join(["---"] * num_cols) + " |"
            lines.insert(1, separator)
        
        return "\n".join(lines)

    def _smart_chunk(self, chunks: List[Dict], max_chunk_size: int = 1000) -> List[Dict]:
        """
        Smart chunking that respects content boundaries.
        """
        smart_chunks = []
        current_chunk = {"text": "", "metadata": {}}
        
        for chunk in chunks:
            chunk_text = chunk['text']
            
            # Don't split tables or code blocks
            if chunk['type'] in ['table', 'code', 'equation']:
                # If current chunk has content, save it
                if current_chunk['text']:
                    smart_chunks.append(current_chunk)
                    current_chunk = {"text": "", "metadata": {}}
                
                # Add special content as its own chunk with metadata
                smart_chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'page': chunk['page'],
                        'type': chunk['type'],
                        'source': chunk['source'],
                        'char_count': len(chunk_text),
                        'token_count': len(chunk_text) // 4,  # Rough estimate: 1 token ≈ 4 chars
                        'char_count_raw': len(chunk_text)
                    }
                })
                continue
            
            # For regular text, combine until max size
            if len(current_chunk['text']) + len(chunk_text) < max_chunk_size:
                current_chunk['text'] += "\n\n" + chunk_text if current_chunk['text'] else chunk_text
                current_chunk['metadata'] = {
                    'page': chunk['page'],
                    'type': 'text',
                    'source': chunk['source'],
                    'char_count': len(current_chunk['text']),
                    'token_count': len(current_chunk['text']) // 4,
                    'char_count_raw': len(current_chunk['text'])
                }
            else:
                # Save current chunk and start new one
                if current_chunk['text']:
                    smart_chunks.append(current_chunk)
                current_chunk = {
                    'text': chunk_text,
                    'metadata': {
                        'page': chunk['page'],
                        'type': 'text',
                        'source': chunk['source'],
                        'char_count': len(chunk_text),
                        'token_count': len(chunk_text) // 4,
                        'char_count_raw': len(chunk_text)
                    }
                }
        
        # Add last chunk
        if current_chunk['text']:
            smart_chunks.append(current_chunk)
        
        return smart_chunks

    def ingest_file(self, file_path: str) -> str:
        """
        Reads a PDF with enhanced processing and stores in ChromaDB.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract with layout preservation
        raw_chunks = self._extract_with_layout(file_path)
        
        # Smart chunking
        processed_chunks = self._smart_chunk(raw_chunks)
        
        # Prepare data for ChromaDB
        texts = [chunk['text'] for chunk in processed_chunks]
        metadatas = [chunk['metadata'] for chunk in processed_chunks]
        ids = [f"chunk_{i}" for i in range(len(texts))]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to database
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return f"Successfully processed {len(texts)} chunks from {os.path.basename(file_path)}"

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[str], List[Dict]]:
        """
        Retrieves the most relevant text chunks with metadata and confidence scores.
        Implementation of Two-Stage Retrieval (Vector Search + Reranking) inspired by LightRAG.
        Returns: (documents, metadatas) where metadatas includes 'confidence' field.
        """
        # 1. Fetch more candidates than needed (e.g., 3x top_k) for the reranker to screen
        initial_k = top_k * 3
        
        # Convert query to embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search the database
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=initial_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results or not results['documents'] or not results['documents'][0]:
            return [], []

        flat_docs = results['documents'][0]
        flat_metas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(flat_docs)
        
        # 2. Reranking Step (The LightRAG improvement)
        # Create pairs of [query, doc] for the Cross-Encoder
        pairs = [[query, doc] for doc in flat_docs]
        
        # Predict scores (higher is better)
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine docs, metas, and scores
        scored_results = []
        for doc, meta, score in zip(flat_docs, flat_metas, rerank_scores):
            # Normalize score to 0-1 range roughly (sigmoid)
            # CrossEncoder scores are logits, usually between -10 and 10
            import math
            confidence = 1 / (1 + math.exp(-score))
            
            enriched_meta = dict(meta) if meta else {}
            enriched_meta['confidence'] = round(confidence, 3)
            enriched_meta['rerank_score'] = float(score) # Store raw score for debugging
            
            scored_results.append({
                'doc': doc,
                'meta': enriched_meta,
                'score': score
            })
            
        # Sort by the new reranker score (descending)
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Slice to the requested top_k
        final_results = scored_results[:top_k]
        
        final_docs = [item['doc'] for item in final_results]
        final_metas = [item['meta'] for item in final_results]
            
        return final_docs, final_metas
    
    def analyze_query(self, query: str) -> dict:
        """
        Analyze the query to extract search terms and generate query rewrite.
        Returns reasoning trace for transparency.
        """
        import re
        
        # Extract key words (remove stopwords and short words)
        stopwords = {'what', 'is', 'the', 'a', 'an', 'how', 'does', 'do', 'can', 'could', 
                     'would', 'should', 'will', 'are', 'was', 'were', 'been', 'being',
                     'have', 'has', 'had', 'of', 'to', 'for', 'in', 'on', 'with', 'by',
                     'about', 'this', 'that', 'these', 'those', 'it', 'its', 'and', 'or'}
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        search_terms = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Generate query rewrite - simplified expansion
        query_rewrite = query.strip()
        if query_rewrite.endswith('?'):
            query_rewrite = query_rewrite[:-1]
        
        return {
            'query_rewrite': query_rewrite,
            'search_terms': search_terms[:6],  # Limit to 6 terms
            'retrieval_strategy': f"Semantic search using local embeddings (all-MiniLM-L6-v2)"
        }
    
RAGEngine = RAGEngine