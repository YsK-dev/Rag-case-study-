import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_engine import RAGEngine
from llm_client import OllamaClient


def main():
    print("=" * 50)
    print(" RAG Demo - Document Q&A System")
    print("=" * 50)
    print()
    
    # Initialize components
    print("Initializing RAG Engine...")
    rag = RAGEngine(db_path="./chroma_db", collection_name="demo_docs")
    
    print("Initializing Ollama LLM (qwen3:1.7b)...")
    llm = OllamaClient(model="qwen3:1.7b")
    print()
    
    # Check if we need to ingest documents
    print("Checking for documents...")
    sample_pdf = "data/sample.pdf"
    
    if os.path.exists(sample_pdf):
        print(f"Found: {sample_pdf}")
        
        # Check if already ingested (simple check)
        try:
            test_results = rag.retrieve("test", top_k=1)
            if not test_results:
                print("   Ingesting document into vector database...")
                result = rag.ingest_file(sample_pdf)
                print(f"{result}")
            else:
                print("Document already in database")
        except:
            print("   Ingesting document into vector database...")
            result = rag.ingest_file(sample_pdf)
            print(f"{result}")
    else:
        print(f"No sample PDF found at {sample_pdf}")
        print("You can still ask general questions without RAG context.")
    
    print()
    print("=" * 50)
    print("Interactive Q&A Mode")
    print("=" * 50)
    print("Ask questions about the document, or type 'quit' to exit.")
    print()
    
    # Interactive loop
    while True:
        try:
            # Get user question
            question = input("Your question: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            # Retrieve relevant context
            print("\nSearching for relevant information...")
            context_chunks = rag.retrieve(question, top_k=3)
            
            if context_chunks:
                print(f"Found {len(context_chunks)} relevant chunks")
                
                # Generate answer using LLM + context
                print("Generating answer...")
                answer = llm.generate_answer(question, context_chunks)
                
                print("\n" + "=" * 50)
                print("Answer:")
                print("=" * 50)
                print(answer)
                print("=" * 50)
                print()
                
                # Optionally show sources
                show_sources = input("Show source chunks? (y/n): ").strip().lower()
                if show_sources == 'y':
                    print("\nSource Chunks:")
                    for i, chunk in enumerate(context_chunks, 1):
                        print(f"\n--- Chunk {i} ---")
                        print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    print()
            else:
                print("No relevant context found. Asking LLM directly...")
                answer = llm.chat(question)
                print("\nAnswer (without RAG context):")
                print(answer)
                print()
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print()


if __name__ == "__main__":
    main()
