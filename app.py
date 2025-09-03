"""
Streamlit web application for the RAG Assistant.

This module provides a modern web interface for interacting with the RAG assistant,
allowing users to upload documents, ask questions, and view responses.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.rag_chain import RAGChain
from src.vector_store import VectorStoreManager
from src.document_loader import DocumentLoader
from src.utils import setup_logging, create_sample_data, validate_environment
from config import get_settings, validate_config


def setup_page():
    """Configure the Streamlit page."""
    st.set_page_config(
        page_title="RAG Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 RAG Assistant")
    st.markdown("AI-powered question answering from your documents")
    
    # Add custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .source-box {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    .stats-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'vector_store_type' not in st.session_state:
        st.session_state.vector_store_type = "faiss"


def create_rag_chain(documents, vector_store_type="faiss", use_local_model=False):
    """
    Create a RAG chain from documents.
    
    Args:
        documents: List of documents
        vector_store_type: Type of vector store to use
        use_local_model: Whether to use local model instead of OpenAI
        
    Returns:
        Initialized RAG chain or None if failed
    """
    try:
        # Create vector store
        vector_store = VectorStoreManager(store_type=vector_store_type)
        vector_store.add_documents(documents)
        
        # Get settings
        settings = get_settings()
        
        # Create RAG chain
        rag_chain = RAGChain(
            vector_store=vector_store,
            model_name=settings.model_name,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            top_k=settings.top_k,
            enable_memory=settings.enable_memory,
            use_local_model=use_local_model
        )
        
        return rag_chain
        
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        return None


def load_documents_from_directory(directory_path: str):
    """
    Load documents from a directory.
    
    Args:
        directory_path: Path to directory containing documents
        
    Returns:
        List of loaded documents
    """
    try:
        loader = DocumentLoader()
        documents = loader.load_documents(directory_path)
        return documents
    except Exception as e:
        st.error(f"Error loading documents: {e}")
        return []


def load_uploaded_files(uploaded_files):
    """
    Load documents from uploaded files.
    
    Args:
        uploaded_files: List of uploaded file objects
        
    Returns:
        List of loaded documents
    """
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Load documents
            loader = DocumentLoader()
            documents = loader.load_documents(temp_dir)
            return documents
            
    except Exception as e:
        st.error(f"Error loading uploaded files: {e}")
        return []


def display_conversation_history():
    """Display the conversation history."""
    if st.session_state.conversation_history:
        st.subheader("💬 Conversation History")
        
        for i, (question, answer, sources) in enumerate(st.session_state.conversation_history):
            with st.expander(f"Q{i+1}: {question[:50]}...", expanded=False):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")
                
                if sources:
                    st.markdown("**Sources:**")
                    for j, source in enumerate(sources):
                        st.markdown(f"{j+1}. {source}")


def display_system_stats():
    """Display system statistics."""
    if st.session_state.rag_chain:
        st.subheader("📊 System Statistics")
        
        # Get chain info
        chain_info = st.session_state.rag_chain.get_chain_info()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model", chain_info['model_name'])
            st.metric("Temperature", chain_info['temperature'])
        
        with col2:
            st.metric("Top K", chain_info['top_k'])
            st.metric("Memory", "Enabled" if chain_info['memory_enabled'] else "Disabled")
        
        with col3:
            vs_stats = chain_info['vector_store_stats']
            st.metric("Vector Store", vs_stats['store_type'].upper())
            st.metric("Documents", vs_stats['document_count'])
        
        # Memory stats if available
        if st.session_state.rag_chain.memory:
            memory_stats = st.session_state.rag_chain.memory.get_memory_stats()
            st.markdown(f"**Session ID:** {memory_stats['session_id']}")
            st.markdown(f"**Conversation Turns:** {memory_stats['current_turns']}")


def main():
    """Main application function."""
    setup_page()
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        use_local_model = st.checkbox(
            "Use Local Model (No API Key Required)",
            value=False,
            help="Check this to use a local Hugging Face model instead of OpenAI"
        )
        
        # Vector store selection
        vector_store_type = st.selectbox(
            "Vector Store Type",
            ["faiss", "chroma"],
            index=0 if st.session_state.vector_store_type == "faiss" else 1
        )
        st.session_state.vector_store_type = vector_store_type
        
        # Document loading section
        st.header("📚 Document Loading")
        
        load_option = st.radio(
            "Choose document loading method:",
            ["Upload Files", "Use Sample Data", "Load from Directory"]
        )
        
        if load_option == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=["txt", "pdf", "docx", "md"],
                accept_multiple_files=True,
                help="Supported formats: TXT, PDF, DOCX, MD"
            )
            
            if uploaded_files and st.button("Load Uploaded Files"):
                with st.spinner("Loading documents..."):
                    documents = load_uploaded_files(uploaded_files)
                    if documents:
                        st.session_state.rag_chain = create_rag_chain(documents, vector_store_type, use_local_model)
                        st.session_state.documents_loaded = True
                        st.success(f"✅ Loaded {len(documents)} documents!")
                        st.rerun()
        
        elif load_option == "Use Sample Data":
            if st.button("Load Sample Data"):
                with st.spinner("Creating sample data..."):
                    try:
                        # Create sample data
                        create_sample_data("temp_sample_data")
                        documents = load_documents_from_directory("temp_sample_data")
                        
                        if documents:
                            st.session_state.rag_chain = create_rag_chain(documents, vector_store_type, use_local_model)
                            if st.session_state.rag_chain:
                                st.session_state.documents_loaded = True
                                st.success(f"✅ Loaded {len(documents)} sample documents!")
                                
                                # Clean up
                                shutil.rmtree("temp_sample_data", ignore_errors=True)
                                st.rerun()
                            else:
                                st.error("❌ Failed to create RAG chain. Check your configuration.")
                        else:
                            st.error("❌ Failed to load sample documents.")
                    except Exception as e:
                        st.error(f"❌ Error loading sample data: {e}")
                        st.info("💡 Make sure you have set your OPENAI_API_KEY in a .env file")
        
        elif load_option == "Load from Directory":
            directory_path = st.text_input(
                "Directory path:",
                value="data",
                help="Path to directory containing documents"
            )
            
            if st.button("Load from Directory"):
                if os.path.exists(directory_path):
                    with st.spinner("Loading documents..."):
                        documents = load_documents_from_directory(directory_path)
                        if documents:
                            st.session_state.rag_chain = create_rag_chain(documents, vector_store_type, use_local_model)
                            st.session_state.documents_loaded = True
                            st.success(f"✅ Loaded {len(documents)} documents!")
                            st.rerun()
                else:
                    st.error(f"Directory not found: {directory_path}")
        
        # Configuration validation
        if st.button("Validate Configuration"):
            if validate_config():
                st.success("✅ Configuration is valid!")
            else:
                st.error("❌ Configuration validation failed!")
        
        # Environment validation
        if st.button("Check Environment"):
            env_status = validate_environment()
            st.json(env_status)
    
    # Main content area
    if not st.session_state.documents_loaded:
        st.info("👈 Please load documents using the sidebar to get started!")
        
        # Show sample questions
        st.subheader("💡 Example Questions")
        st.markdown("""
        Once you load documents, you can ask questions like:
        - What is the main topic of this document?
        - What are the key findings or conclusions?
        - What methodology was used in this research?
        - What are the limitations mentioned?
        - Can you summarize the main points?
        """)
        
        # Show supported formats
        st.subheader("📋 Supported Document Formats")
        st.markdown("""
        - **Text files** (.txt, .md)
        - **PDF documents** (.pdf)
        - **Word documents** (.docx)
        """)
        
    else:
        # Chat interface
        st.subheader("💬 Ask a Question")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about your documents?",
            key="question_input"
        )
        
        # Process question
        if question and st.button("Ask", type="primary"):
            if st.session_state.rag_chain:
                with st.spinner("🔍 Processing your question..."):
                    try:
                        response = st.session_state.rag_chain.query(question)
                        
                        # Display answer
                        st.markdown("### 🤖 Answer")
                        st.markdown(f'<div class="response-box">{response.answer}</div>', unsafe_allow_html=True)
                        
                        # Display sources
                        if response.retrieved_documents:
                            st.markdown("### 📚 Sources")
                            for i, doc in enumerate(response.retrieved_documents, 1):
                                source = doc.metadata.get('source', 'Unknown')
                                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                
                                with st.expander(f"Source {i}: {os.path.basename(source)}", expanded=False):
                                    st.markdown(f"**File:** {source}")
                                    st.markdown(f"**Content:** {content_preview}")
                        
                        # Add to conversation history
                        sources = [doc.metadata.get('source', 'Unknown') for doc in response.retrieved_documents]
                        st.session_state.conversation_history.append((question, response.answer, sources))
                        
                        # Clear input
                        st.session_state.question_input = ""
                        
                    except Exception as e:
                        st.error(f"❌ Error processing question: {e}")
                        st.info("💡 This might be due to a missing OpenAI API key. Check your .env file.")
            else:
                st.error("❌ RAG chain not initialized. Please load documents first.")
                st.info("💡 Make sure you have set your OPENAI_API_KEY in a .env file")
        
        # Display conversation history
        if st.session_state.conversation_history:
            display_conversation_history()
        
        # Display system stats
        display_system_stats()
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history.clear()
            if st.session_state.rag_chain and st.session_state.rag_chain.memory:
                st.session_state.rag_chain.memory.clear_memory()
            st.success("Conversation history cleared!")
            st.rerun()


if __name__ == "__main__":
    main()
