import os
import chromadb
from typing import List
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

class RAGEngine:
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "docs_collection"):
        """
        Initializes the RAG Engine with a Vector DB and an Embedding Model.
        """
        # 1. Initialize Vector DB (ChromaDB)
        # persistent=True means data stays on disk even after you restart the app.
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        
        # 2. Create or get a collection (think of it like a SQL table)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        
        # 3. Initialize Embedding Model
        # 'all-MiniLM-L6-v2' is small, fast, and runs great on local CPU.
        print("Loading embedding model... (this happens only once)")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded.")

    def ingest_file(self, file_path: str) -> str:
        """
        Reads a PDF, splits it into chunks, and stores embeddings in ChromaDB.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # A. Read PDF
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"

        # B. Simple Chunking (Splitting text)
        # We split by 1000 characters for simplicity.
        # In a pro app, you might use overlap (e.g., recursive splitting).
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Prepare data for ChromaDB
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path} for _ in range(len(chunks))]
        
        # C. Embed and Store
        # We generate embeddings for the text chunks
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # Add to database
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return f"Successfully processed {len(chunks)} chunks from {file_path}"

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Takes a question, finds the most relevant text chunks.
        """
        # 1. Convert the query (question) into numbers (embedding)
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # 2. Search the database for the closest chunks
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Chroma returns a list of lists, we just want the first list of documents
        if results and results['documents']:
            return results['documents'][0]
        return []

# Quick Test (If you run this file directly)
if __name__ == "__main__":
    # Ensure you have a 'test.pdf' in the root or change path to test
    engine = RAGEngine()
    # engine.ingest_file("data/docs/sample.pdf") 
    # results = engine.retrieve("What is this document about?")
    # print(results)