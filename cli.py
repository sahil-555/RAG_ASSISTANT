#!/usr/bin/env python3
"""
Command-line interface for the RAG Assistant.

This module provides a CLI for interacting with the RAG assistant,
allowing users to ask questions and get responses from the knowledge base.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_chain import RAGChain, create_sample_rag_chain
from src.vector_store import VectorStoreManager
from src.document_loader import DocumentLoader
from src.utils import setup_logging, create_sample_data, validate_environment
from config import get_settings, validate_config


def setup_cli_logging():
    """Set up logging for CLI."""
    return setup_logging(
        level="INFO",
        log_file="logs/cli.log",
        log_format="%(asctime)s - %(levelname)s - %(message)s"
    )


def create_rag_chain_from_documents(data_dir: str, vector_store_type: str = "faiss") -> RAGChain:
    """
    Create a RAG chain from documents in a directory.
    
    Args:
        data_dir: Directory containing documents
        vector_store_type: Type of vector store to use
        
    Returns:
        Initialized RAG chain
    """
    logger = setup_cli_logging()
    
    try:
        # Load documents
        logger.info(f"Loading documents from {data_dir}")
        loader = DocumentLoader()
        documents = loader.load_documents(data_dir)
        
        if not documents:
            logger.warning(f"No documents found in {data_dir}")
            return None
        
        # Create vector store
        logger.info(f"Creating {vector_store_type} vector store")
        vector_store = VectorStoreManager(store_type=vector_store_type)
        vector_store.add_documents(documents)
        
        # Create RAG chain
        settings = get_settings()
        rag_chain = RAGChain(
            vector_store=vector_store,
            model_name=settings.model_name,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            top_k=settings.top_k,
            enable_memory=settings.enable_memory
        )
        
        logger.info(f"RAG chain created successfully with {len(documents)} documents")
        return rag_chain
        
    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}")
        return None


def interactive_mode(rag_chain: RAGChain):
    """
    Run the RAG assistant in interactive mode.
    
    Args:
        rag_chain: Initialized RAG chain
    """
    print("\n🤖 RAG Assistant - Interactive Mode")
    print("=" * 50)
    print("Type 'quit', 'exit', or 'q' to end the session")
    print("Type 'help' for available commands")
    print("Type 'stats' to see system statistics")
    print("=" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n❓ Your question: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Thanks for using RAG Assistant.")
                break
            
            elif user_input.lower() == 'help':
                show_help()
                continue
            
            elif user_input.lower() == 'stats':
                show_stats(rag_chain)
                continue
            
            elif user_input.lower() == 'clear':
                if rag_chain.memory:
                    rag_chain.memory.clear_memory()
                    print("🧹 Conversation memory cleared.")
                continue
            
            # Process the question
            print("\n🔍 Processing your question...")
            response = rag_chain.query(user_input)
            
            # Display response
            print(f"\n🤖 Answer: {response.answer}")
            print(f"\n📚 Sources: {len(response.retrieved_documents)} documents retrieved")
            
            # Show source information
            if response.retrieved_documents:
                print("\n📖 Source documents:")
                for i, doc in enumerate(response.retrieved_documents, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    print(f"  {i}. {source}")
                    print(f"     {content_preview}")
            
            print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using RAG Assistant.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'help' for assistance.")


def single_query_mode(rag_chain: RAGChain, question: str):
    """
    Run a single query and display the result.
    
    Args:
        rag_chain: Initialized RAG chain
        question: User question
    """
    print(f"\n🔍 Processing question: {question}")
    print("=" * 50)
    
    try:
        response = rag_chain.query(question)
        
        print(f"\n🤖 Answer: {response.answer}")
        print(f"\n📚 Sources: {len(response.retrieved_documents)} documents retrieved")
        
        # Show source information
        if response.retrieved_documents:
            print("\n📖 Source documents:")
            for i, doc in enumerate(response.retrieved_documents, 1):
                source = doc.metadata.get('source', 'Unknown')
                content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"  {i}. {source}")
                print(f"     {content_preview}")
        
        # Show metadata
        print(f"\n🔧 Response metadata:")
        for key, value in response.metadata.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"\n❌ Error processing question: {e}")


def show_help():
    """Display help information."""
    help_text = """
📚 RAG Assistant Help

Available Commands:
  help          - Show this help message
  stats         - Display system statistics
  clear         - Clear conversation memory
  quit/exit/q   - Exit the assistant

Usage Tips:
  - Ask specific questions for better results
  - The assistant uses your document knowledge base
  - Responses are based on retrieved context
  - Memory is maintained across conversation turns

Example Questions:
  - "What is the main topic of this document?"
  - "What are the key findings?"
  - "Can you summarize the methodology?"
  - "What are the limitations mentioned?"
"""
    print(help_text)


def show_stats(rag_chain: RAGChain):
    """Display system statistics."""
    print("\n📊 System Statistics")
    print("=" * 30)
    
    # RAG chain info
    chain_info = rag_chain.get_chain_info()
    print(f"Model: {chain_info['model_name']}")
    print(f"Temperature: {chain_info['temperature']}")
    print(f"Top K: {chain_info['top_k']}")
    print(f"Memory Enabled: {chain_info['memory_enabled']}")
    
    # Vector store stats
    vs_stats = chain_info['vector_store_stats']
    print(f"Vector Store: {vs_stats['store_type']}")
    print(f"Documents: {vs_stats['document_count']}")
    print(f"Embedding Model: {vs_stats['embedding_model']}")
    
    # Memory stats if available
    if rag_chain.memory:
        memory_stats = rag_chain.memory.get_memory_stats()
        print(f"Session ID: {memory_stats['session_id']}")
        print(f"Conversation Turns: {memory_stats['current_turns']}")
        print(f"Memory Usage: {memory_stats['memory_usage_percent']:.1f}%")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Assistant - AI-powered question answering from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with sample data
  python cli.py --interactive --sample
  
  # Single query with custom data
  python cli.py --query "What is this about?" --data-dir ./my_documents
  
  # Use specific vector store type
  python cli.py --interactive --vector-store chroma --data-dir ./data
        """
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single question to ask"
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="data",
        help="Directory containing documents (default: data)"
    )
    
    parser.add_argument(
        "--vector-store", "-v",
        type=str,
        choices=["faiss", "chroma"],
        default="faiss",
        help="Vector store type to use (default: faiss)"
    )
    
    parser.add_argument(
        "--sample", "-s",
        action="store_true",
        help="Use sample data if no documents found"
    )
    
    parser.add_argument(
        "--config", "-c",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    args = parser.parse_args()
    
    # Validate configuration if requested
    if args.config:
        print("🔧 Validating configuration...")
        if validate_config():
            print("✅ Configuration is valid!")
            return 0
        else:
            print("❌ Configuration validation failed!")
            return 1
    
    # Check if we have a query or interactive mode
    if not args.interactive and not args.query:
        parser.print_help()
        return 1
    
    # Validate environment
    print("🔍 Validating environment...")
    env_status = validate_environment()
    
    if not env_status.get("python_version", False):
        print("❌ Python 3.8+ is required")
        return 1
    
    if not env_status.get("openai_api_key", False):
        print("❌ OPENAI_API_KEY environment variable is required")
        print("Please set your OpenAI API key and try again")
        return 1
    
    print("✅ Environment validation passed")
    
    # Create data directory if it doesn't exist
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        if args.sample:
            print(f"📁 Creating sample data in {data_dir}")
            create_sample_data(str(data_dir))
        else:
            print(f"❌ Data directory {data_dir} not found")
            print("Use --sample to create sample data or specify a valid --data-dir")
            return 1
    
    # Create RAG chain
    print("🚀 Initializing RAG Assistant...")
    rag_chain = create_rag_chain_from_documents(str(data_dir), args.vector_store)
    
    if not rag_chain:
        print("❌ Failed to create RAG chain")
        return 1
    
    print("✅ RAG Assistant initialized successfully!")
    
    # Run in appropriate mode
    if args.interactive:
        interactive_mode(rag_chain)
    elif args.query:
        single_query_mode(rag_chain, args.query)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
