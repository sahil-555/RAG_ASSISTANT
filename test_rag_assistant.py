#!/usr/bin/env python3
"""
Comprehensive test script for the RAG Assistant.

This script demonstrates all the major features and capabilities
of the RAG assistant system.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_chain import RAGChain, create_sample_rag_chain
from src.vector_store import VectorStoreManager, create_sample_vector_store
from src.document_loader import DocumentLoader, create_sample_documents
from src.memory import ConversationMemory, PersistentMemory
from src.utils import setup_logging, create_sample_data, validate_environment
from config import get_settings, validate_config


def test_document_loader():
    """Test the document loader functionality."""
    print("📚 Testing Document Loader")
    print("=" * 40)
    
    try:
        # Create sample data
        create_sample_data("test_data")
        
        # Test document loading
        loader = DocumentLoader()
        documents = loader.load_documents("test_data")
        
        print(f"✅ Loaded {len(documents)} documents")
        print(f"✅ Supported formats: {loader.get_supported_formats()}")
        
        # Test individual document loading
        for doc in documents[:2]:  # Show first 2
            print(f"  - {doc.metadata.get('source', 'Unknown')}: {len(doc.page_content)} chars")
        
        # Clean up
        import shutil
        shutil.rmtree("test_data", ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Document loader test failed: {e}")
        return False


def test_vector_store():
    """Test the vector store functionality."""
    print("\n🔍 Testing Vector Store")
    print("=" * 40)
    
    try:
        # Test FAISS
        print("Testing FAISS vector store...")
        faiss_store = create_sample_vector_store("faiss")
        print(f"✅ FAISS store created with {faiss_store.get_stats()['document_count']} documents")
        
        # Test search
        query = "What is machine learning?"
        results = faiss_store.similarity_search(query, k=2)
        print(f"✅ FAISS search returned {len(results)} results")
        
        # Test Chroma
        print("\nTesting Chroma vector store...")
        chroma_store = create_sample_vector_store("chroma")
        print(f"✅ Chroma store created with {chroma_store.get_stats()['document_count']} documents")
        
        # Test search
        results = chroma_store.similarity_search(query, k=2)
        print(f"✅ Chroma search returned {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False


def test_memory():
    """Test the conversation memory functionality."""
    print("\n🧠 Testing Conversation Memory")
    print("=" * 40)
    
    try:
        # Test basic memory
        memory = ConversationMemory(max_turns=5, context_window=3)
        
        # Add interactions
        interactions = [
            ("What is AI?", "AI is artificial intelligence...", 2),
            ("How does it work?", "AI works by processing data...", 3),
            ("What are the benefits?", "AI provides automation...", 1),
        ]
        
        for query, response, docs in interactions:
            memory.add_interaction(query, response, docs)
        
        print(f"✅ Added {len(interactions)} interactions to memory")
        
        # Test context addition
        current_query = "Can you summarize our conversation?"
        enhanced_query = memory.add_context(current_query)
        print(f"✅ Enhanced query length: {len(enhanced_query)} chars")
        
        # Test memory stats
        stats = memory.get_memory_stats()
        print(f"✅ Memory stats: {stats}")
        
        # Test persistent memory
        persistent_memory = PersistentMemory(storage_path="test_memory")
        persistent_memory.add_interaction("Test query", "Test response", 1)
        print("✅ Persistent memory test passed")
        
        # Clean up
        import shutil
        shutil.rmtree("test_memory", ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


def test_rag_chain():
    """Test the RAG chain functionality."""
    print("\n🤖 Testing RAG Chain")
    print("=" * 40)
    
    try:
        # Create sample RAG chain
        rag_chain = create_sample_rag_chain("faiss")
        print("✅ RAG chain created successfully")
        
        # Test single query
        test_questions = [
            "What is artificial intelligence?",
            "What are the types of machine learning?",
            "How does deep learning work?"
        ]
        
        for question in test_questions:
            print(f"\nTesting question: {question}")
            response = rag_chain.query(question)
            print(f"✅ Response length: {len(response.answer)} chars")
            print(f"✅ Documents retrieved: {len(response.retrieved_documents)}")
        
        # Test batch query
        print("\nTesting batch query...")
        batch_responses = rag_chain.batch_query(test_questions[:2])
        print(f"✅ Batch query returned {len(batch_responses)} responses")
        
        # Test chain info
        chain_info = rag_chain.get_chain_info()
        print(f"✅ Chain info: {chain_info['model_name']}, {chain_info['top_k']}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG chain test failed: {e}")
        return False


def test_configuration():
    """Test the configuration system."""
    print("\n🔧 Testing Configuration")
    print("=" * 40)
    
    try:
        # Test settings
        settings = get_settings()
        print(f"✅ Model: {settings.model_name}")
        print(f"✅ Vector store: {settings.vector_store_type}")
        print(f"✅ Top K: {settings.top_k}")
        
        # Test validation
        if validate_config():
            print("✅ Configuration validation passed")
        else:
            print("⚠️ Configuration validation failed (expected without API key)")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\n🔧 Testing Utilities")
    print("=" * 40)
    
    try:
        from src.utils import clean_text, truncate_text, extract_keywords
        
        # Test text cleaning
        test_text = "  This   is   a   test   text!  "
        cleaned = clean_text(test_text)
        print(f"✅ Text cleaning: '{test_text}' -> '{cleaned}'")
        
        # Test text truncation
        truncated = truncate_text("This is a very long text that should be truncated", 20)
        print(f"✅ Text truncation: {truncated}")
        
        # Test keyword extraction
        keywords = extract_keywords("Machine learning is a subset of artificial intelligence")
        print(f"✅ Keywords extracted: {keywords}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        return False


def run_comprehensive_test():
    """Run all tests and provide a summary."""
    print("🚀 RAG Assistant Comprehensive Test Suite")
    print("=" * 60)
    
    # Set up logging
    logger = setup_logging(level="INFO")
    
    # Test results
    test_results = {}
    
    # Run tests
    test_results['document_loader'] = test_document_loader()
    test_results['vector_store'] = test_vector_store()
    test_results['memory'] = test_memory()
    test_results['rag_chain'] = test_rag_chain()
    test_results['configuration'] = test_configuration()
    test_results['utilities'] = test_utilities()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The RAG Assistant is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        return False


def main():
    """Main test function."""
    try:
        success = run_comprehensive_test()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⏹️ Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
