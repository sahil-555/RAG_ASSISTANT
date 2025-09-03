"""
Vector store management for the RAG Assistant project.

This module handles creating, managing, and querying vector stores for
efficient document retrieval using semantic similarity.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

import numpy as np
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.vectorstores.base import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Perform similarity search."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store implementation."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_model: HuggingFace model name for embeddings
        """
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store: Optional[FAISS] = None
        self.documents: List[Document] = []
        
        logger.info(f"Initialized FAISS vector store with {embedding_model}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the FAISS vector store."""
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        self.documents.extend(documents)
        
        if self.vector_store is None:
            # Create new vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Created new FAISS vector store with {len(documents)} documents")
        else:
            # Add to existing vector store
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to existing FAISS vector store")
    
    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Perform similarity search using FAISS."""
        if self.vector_store is None:
            logger.warning("Vector store is empty. No documents to search.")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k, **kwargs)
            logger.info(f"FAISS search returned {len(results)} results for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error in FAISS similarity search: {e}")
            return []
    
    def save(self, path: str) -> None:
        """Save the FAISS vector store to disk."""
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save FAISS index
            self.vector_store.save_local(path)
            
            # Save documents separately
            docs_path = f"{path}_documents.pkl"
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info(f"FAISS vector store saved to {path}")
        except Exception as e:
            logger.error(f"Error saving FAISS vector store: {e}")
            raise
    
    def load(self, path: str) -> None:
        """Load the FAISS vector store from disk."""
        try:
            # Load FAISS index
            self.vector_store = FAISS.load_local(path, self.embeddings)
            
            # Load documents
            docs_path = f"{path}_documents.pkl"
            if os.path.exists(docs_path):
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
            else:
                logger.warning(f"Documents file not found at {docs_path}")
                self.documents = []
            
            logger.info(f"FAISS vector store loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading FAISS vector store: {e}")
            raise


class ChromaVectorStore(BaseVectorStore):
    """Chroma-based vector store implementation."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Chroma vector store.
        
        Args:
            embedding_model: HuggingFace model name for embeddings
        """
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store: Optional[Chroma] = None
        self.documents: List[Document] = []
        
        logger.info(f"Initialized Chroma vector store with {embedding_model}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the Chroma vector store."""
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        self.documents.extend(documents)
        
        if self.vector_store is None:
            # Create new vector store
            self.vector_store = Chroma.from_documents(documents, self.embeddings)
            logger.info(f"Created new Chroma vector store with {len(documents)} documents")
        else:
            # Add to existing vector store
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to existing Chroma vector store")
    
    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Perform similarity search using Chroma."""
        if self.vector_store is None:
            logger.warning("Vector store is empty. No documents to search.")
            return []
        
        try:
            results = self.vector_store.similarity_search(query, k=k, **kwargs)
            logger.info(f"Chroma search returned {len(results)} results for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error in Chroma similarity search: {e}")
            return []
    
    def save(self, path: str) -> None:
        """Save the Chroma vector store to disk."""
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Chroma persists automatically, but we can save documents separately
            docs_path = f"{path}_documents.pkl"
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info(f"Chroma vector store documents saved to {docs_path}")
        except Exception as e:
            logger.error(f"Error saving Chroma vector store: {e}")
            raise
    
    def load(self, path: str) -> None:
        """Load the Chroma vector store from disk."""
        try:
            # Chroma loads automatically from the path
            self.vector_store = Chroma(persist_directory=path, embedding_function=self.embeddings)
            
            # Load documents
            docs_path = f"{path}_documents.pkl"
            if os.path.exists(docs_path):
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
            else:
                logger.warning(f"Documents file not found at {docs_path}")
                self.documents = []
            
            logger.info(f"Chroma vector store loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading Chroma vector store: {e}")
            raise


class VectorStoreManager:
    """Main vector store manager that handles different vector store types."""
    
    def __init__(self, store_type: str = "faiss", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the vector store manager.
        
        Args:
            store_type: Type of vector store ("faiss" or "chroma")
            embedding_model: HuggingFace model name for embeddings
        """
        self.store_type = store_type.lower()
        self.embedding_model = embedding_model
        
        if self.store_type == "faiss":
            self.vector_store = FAISSVectorStore(embedding_model)
        elif self.store_type == "chroma":
            self.vector_store = ChromaVectorStore(embedding_model)
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}. Use 'faiss' or 'chroma'")
        
        logger.info(f"Initialized {self.store_type.upper()} vector store manager")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        self.vector_store.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Document]:
        """Perform similarity search."""
        return self.vector_store.similarity_search(query, k=k, **kwargs)
    
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        self.vector_store.save(path)
    
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        self.vector_store.load(path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "store_type": self.store_type,
            "embedding_model": self.embedding_model,
            "document_count": len(self.vector_store.documents),
            "vector_store_initialized": self.vector_store.vector_store is not None
        }
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self.vector_store.documents = []
        self.vector_store.vector_store = None
        logger.info("Vector store cleared")


def create_sample_vector_store(store_type: str = "faiss") -> VectorStoreManager:
    """Create a sample vector store with sample documents for testing."""
    from .document_loader import create_sample_documents
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Create vector store
    vector_store = VectorStoreManager(store_type=store_type)
    vector_store.add_documents(documents)
    
    return vector_store


if __name__ == "__main__":
    # Test the vector store
    print("🔍 Vector Store Test")
    print("=" * 40)
    
    # Test FAISS
    print("\nTesting FAISS Vector Store:")
    faiss_store = create_sample_vector_store("faiss")
    print(f"Stats: {faiss_store.get_stats()}")
    
    # Test search
    query = "What is machine learning?"
    results = faiss_store.similarity_search(query, k=2)
    print(f"\nSearch results for '{query}':")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content[:100]}...")
    
    # Test Chroma
    print("\nTesting Chroma Vector Store:")
    chroma_store = create_sample_vector_store("chroma")
    print(f"Stats: {chroma_store.get_stats()}")
    
    # Test search
    results = chroma_store.similarity_search(query, k=2)
    print(f"\nSearch results for '{query}':")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content[:100]}...")
