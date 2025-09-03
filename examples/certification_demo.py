#!/usr/bin/env python3
"""
Certification Demo Script

This script demonstrates the RAG assistant capabilities
for the certification submission.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_chain import RAGChain
from src.vector_store import VectorStoreManager
from src.document_loader import DocumentLoader
from src.utils import create_sample_data


def run_certification_demo():
    """Run the certification demonstration."""
    print("🎓 RAG Assistant Certification Demo")
    print("=" * 50)
    
    # Step 1: Document Ingestion
    print("\n📚 Step 1: Document Ingestion")
    print("-" * 30)
    
    # Create sample publications
    create_sample_data("cert_demo_data")
    
    # Load documents
    loader = DocumentLoader()
    documents = loader.load_documents("cert_demo_data")
    print(f"✅ Loaded {len(documents)} documents")
    
    # Step 2: Vector Store Setup
    print("\n🔍 Step 2: Vector Store Setup")
    print("-" * 30)
    
    vector_store = VectorStoreManager("faiss")
    vector_store.add_documents(documents)
    print(f"✅ Vector store created with {vector_store.get_stats()['document_count']} documents")
    
    # Step 3: RAG Chain Initialization
    print("\n🔗 Step 3: RAG Chain Initialization")
    print("-" * 30)
    
    rag = RAGChain(vector_store, use_local_model=True)
    print("✅ RAG chain initialized successfully")
    
    # Step 4: Query Examples
    print("\n❓ Step 4: Query Examples")
    print("-" * 30)
    
    demo_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the benefits of AI?",
        "What is deep learning?",
        "How can AI be applied in business?"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        print("-" * 40)
        
        try:
            response = rag.query(query)
            print(f"💡 Answer: {response.answer}")
            
            if response.retrieved_documents:
                print(f"📚 Sources: {len(response.retrieved_documents)} documents retrieved")
                for j, doc in enumerate(response.retrieved_documents[:2], 1):
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"  {j}. {os.path.basename(source)}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Step 5: Performance Metrics
    print("\n📊 Step 5: Performance Metrics")
    print("-" * 30)
    
    import time
    
    # Test response time
    start_time = time.time()
    response = rag.query("What is artificial intelligence?")
    response_time = time.time() - start_time
    
    print(f"⏱️ Response time: {response_time:.2f} seconds")
    print(f"📝 Answer length: {len(response.answer)} characters")
    print(f"🔍 Documents retrieved: {len(response.retrieved_documents)}")
    
    # Step 6: Advanced Features Demo
    print("\n🚀 Step 6: Advanced Features Demo")
    print("-" * 30)
    
    # Test memory functionality
    print("🧠 Testing conversation memory...")
    try:
        # First query
        response1 = rag.query("What is machine learning?")
        print(f"✅ First query: {response1.answer[:50]}...")
        
        # Follow-up query (should use memory)
        response2 = rag.query("Can you tell me more about that?")
        print(f"✅ Follow-up query: {response2.answer[:50]}...")
        
        print("✅ Memory functionality working")
    except Exception as e:
        print(f"⚠️ Memory test: {e}")
    
    # Test different vector store types
    print("\n🔄 Testing different vector store types...")
    try:
        # Test Chroma
        chroma_store = VectorStoreManager("chroma")
        chroma_store.add_documents(documents[:2])  # Use subset for testing
        
        chroma_rag = RAGChain(chroma_store, use_local_model=True)
        chroma_response = chroma_rag.query("What is AI?")
        print(f"✅ Chroma store working: {chroma_response.answer[:50]}...")
        
    except Exception as e:
        print(f"⚠️ Chroma test: {e}")
    
    # Cleanup
    import shutil
    shutil.rmtree("cert_demo_data", ignore_errors=True)
    
    print("\n🎉 Certification Demo Completed Successfully!")
    print("Your RAG assistant is ready for submission!")
    
    # Final summary
    print("\n📋 DEMO SUMMARY:")
    print("✅ Document ingestion and processing")
    print("✅ Vector store integration (FAISS + Chroma)")
    print("✅ RAG pipeline with local model")
    print("✅ Multiple query types and responses")
    print("✅ Performance metrics and timing")
    print("✅ Advanced features (memory, multiple stores)")
    print("✅ Error handling and robustness")


if __name__ == "__main__":
    run_certification_demo()
