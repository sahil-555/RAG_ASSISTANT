"""
Core RAG (Retrieval-Augmented Generation) chain implementation.

This module implements the main RAG pipeline that combines document retrieval
with LLM-based response generation.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from .local_llm import LocalLLM, SimpleRuleBasedLLM

from .vector_store import VectorStoreManager
from .memory import ConversationMemory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response from the RAG chain."""
    answer: str
    retrieved_documents: List[Document]
    query: str
    metadata: Dict[str, Any]


class RAGChain:
    """Main RAG chain that orchestrates retrieval and generation."""
    
    def __init__(
        self,
        vector_store: VectorStoreManager,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        enable_memory: bool = True,
        use_local_model: bool = False
    ):
        """
        Initialize the RAG chain.
        
        Args:
            vector_store: Vector store manager for document retrieval
            model_name: OpenAI model name
            temperature: Model temperature for response generation
            max_tokens: Maximum tokens for response generation
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score for documents
            enable_memory: Whether to enable conversation memory
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.enable_memory = enable_memory
        
        # Initialize LLM
        if use_local_model:
            try:
                # Try to use local Hugging Face model
                self.llm = LocalLLM(
                    model_name="microsoft/DialoGPT-medium",
                    max_length=max_tokens,
                    temperature=temperature
                )
                logger.info("Using local Hugging Face model")
            except Exception as e:
                logger.warning(f"Failed to load local model: {e}, falling back to simple rule-based model")
                self.llm = SimpleRuleBasedLLM()
                logger.info("Using simple rule-based model")
        else:
            # Use OpenAI
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info("Using OpenAI model")
        
        # Initialize memory if enabled
        self.memory = ConversationMemory() if enable_memory else None
        
        # Store model type for later use
        if hasattr(self.llm, 'model_type'):
            self.model_type = self.llm.model_type
            logger.info(f"Detected model type: {self.model_type}")
        else:
            self.model_type = 'openai'
            logger.info("Using OpenAI model type")
        
        # Create the RAG chain only for OpenAI models
        if self.model_type == 'openai':
            self.chain = self._create_chain()
        else:
            self.chain = None
            logger.info(f"Using {self.model_type} model - no LangChain chain needed")
        
        logger.info(f"Initialized RAG chain with {model_name}")
    
    def _create_chain(self):
        """Create the RAG chain using LangChain."""
        
        # For OpenAI models, use the standard LangChain chain
        prompt_template = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant that answers questions based on the provided context.
        
        Context information:
        {context}
        
        Question: {question}
        
        Instructions:
        1. Answer the question based ONLY on the provided context
        2. If the context doesn't contain enough information to answer the question, say so
        3. Be concise but comprehensive
        4. Use the context to provide accurate and relevant information
        5. If you're unsure about something, acknowledge the uncertainty
        
        Answer:
        """)
        
        # Create the chain
        chain = (
            {"context": self._retrieve_documents, "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _retrieve_documents(self, query: str) -> str:
        """
        Retrieve relevant documents and format them as context.
        
        Args:
            query: User query
            
        Returns:
            Formatted context string from retrieved documents
        """
        try:
            # Retrieve documents
            documents = self.vector_store.similarity_search(query, k=self.top_k)
            
            if not documents:
                return "No relevant documents found for this query."
            
            # Format documents as context
            context_parts = []
            for i, doc in enumerate(documents, 1):
                source = doc.metadata.get('source', 'Unknown source')
                content = doc.page_content.strip()
                context_parts.append(f"Document {i} (Source: {source}):\n{content}\n")
            
            context = "\n".join(context_parts)
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return "Error retrieving documents. Please try again."
    
    def _format_documents_as_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents as context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found for this query."
        
        try:
            # Format documents as context
            context_parts = []
            for i, doc in enumerate(documents, 1):
                source = doc.metadata.get('source', 'Unknown source')
                content = doc.page_content.strip()
                context_parts.append(f"Document {i} (Source: {source}):\n{content}\n")
            
            context = "\n".join(context_parts)
            return context
            
        except Exception as e:
            logger.error(f"Error formatting documents as context: {e}")
            return "Error formatting documents. Please try again."
    
    def query(self, question: str, use_memory: bool = True) -> RAGResponse:
        """
        Process a query through the RAG chain.
        
        Args:
            question: User question
            use_memory: Whether to use conversation memory
            
        Returns:
            RAGResponse object with answer and metadata
        """
        try:
            logger.info(f"Processing query: {question}")
            
            # Add context from memory if enabled
            if use_memory and self.memory:
                question_with_context = self.memory.add_context(question)
            else:
                question_with_context = question
            
            # Get retrieved documents for response
            retrieved_docs = self.vector_store.similarity_search(question, k=self.top_k)
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Generate answer using the chain
            context = self._format_documents_as_context(retrieved_docs)
            logger.info(f"Context length: {len(context)} characters")
            
            if self.model_type == 'openai' and self.chain is not None:
                # Use OpenAI chain
                answer = self.chain.invoke({
                    "context": context,
                    "question": question_with_context
                })
            else:
                # Use local model directly
                prompt = f"""Context information:
{context}

Question: {question_with_context}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information to answer the question, say so
3. Be concise but comprehensive
4. Use the context to provide accurate and relevant information
5. If you're unsure about something, acknowledge the uncertainty

Answer:"""
                
                answer = self.llm.generate(prompt, max_new_tokens=self.max_tokens)
            
            logger.info(f"Generated answer: {answer[:100]}...")
            
            # Update memory if enabled
            if use_memory and self.memory:
                self.memory.add_interaction(question, answer)
            
            # Create response object
            response = RAGResponse(
                answer=answer,
                retrieved_documents=retrieved_docs,
                query=question,
                metadata={
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "top_k": self.top_k,
                    "similarity_threshold": self.similarity_threshold,
                    "memory_enabled": self.enable_memory,
                    "documents_retrieved": len(retrieved_docs)
                }
            )
            
            logger.info(f"Generated response for query: {question[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG chain: {e}")
            # Return error response
            return RAGResponse(
                answer=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                retrieved_documents=[],
                query=question,
                metadata={
                    "error": str(e),
                    "model": self.model_name,
                    "memory_enabled": self.enable_memory
                }
            )
    
    def batch_query(self, questions: List[str]) -> List[RAGResponse]:
        """
        Process multiple queries in batch.
        
        Args:
            questions: List of questions
            
        Returns:
            List of RAGResponse objects
        """
        responses = []
        for question in questions:
            response = self.query(question)
            responses.append(response)
        
        return responses
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get information about the RAG chain configuration."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "memory_enabled": self.enable_memory,
            "vector_store_stats": self.vector_store.get_stats()
        }
    
    def update_config(self, **kwargs):
        """Update RAG chain configuration."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
        
        # Recreate chain if LLM parameters changed
        if any(key in kwargs for key in ['model_name', 'temperature', 'max_tokens']):
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            self.chain = self._create_chain()
            logger.info("Recreated RAG chain with updated configuration")


class AdvancedRAGChain(RAGChain):
    """Advanced RAG chain with additional features like reranking and reasoning."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_reranking = kwargs.get('enable_reranking', False)
        self.enable_reasoning = kwargs.get('enable_reasoning', False)
    
    def _create_chain(self):
        """Create advanced RAG chain with reasoning capabilities."""
        
        if self.enable_reasoning:
            # Chain-of-thought style prompt
            prompt_template = ChatPromptTemplate.from_template("""
            You are a helpful AI assistant that answers questions based on the provided context.
            
            Context information:
            {context}
            
            Question: {question}
            
            Let's approach this step by step:
            1. First, let me analyze the context to understand what information is available
            2. Then, I'll identify the key points relevant to the question
            3. Finally, I'll provide a comprehensive answer based on the context
            
            Analysis:
            """)
        else:
            # Standard prompt
            prompt_template = ChatPromptTemplate.from_template("""
            You are a helpful AI assistant that answers questions based on the provided context.
            
            Context information:
            {context}
            
            Question: {question}
            
            Instructions:
            1. Answer the question based ONLY on the provided context
            2. If the context doesn't contain enough information to answer the question, say so
            3. Be concise but comprehensive
            4. Use the context to provide accurate and relevant information
            5. If you're unsure about something, acknowledge the uncertainty
            
            Answer:
            """)
        
        # Create the chain
        chain = (
            {"context": self._retrieve_documents, "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def query_with_reasoning(self, question: str) -> RAGResponse:
        """Query with explicit reasoning steps."""
        # Temporarily enable reasoning
        original_reasoning = self.enable_reasoning
        self.enable_reasoning = True
        self.chain = self._create_chain()
        
        try:
            response = self.query(question)
        finally:
            # Restore original setting
            self.enable_reasoning = original_reasoning
            self.chain = self._create_chain()
        
        return response


def create_sample_rag_chain(vector_store_type: str = "faiss") -> RAGChain:
    """Create a sample RAG chain for testing purposes."""
    from .vector_store import create_sample_vector_store
    
    # Create sample vector store
    vector_store = create_sample_vector_store(vector_store_type)
    
    # Create RAG chain
    rag_chain = RAGChain(
        vector_store=vector_store,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        top_k=3
    )
    
    return rag_chain


if __name__ == "__main__":
    # Test the RAG chain
    print("🤖 RAG Chain Test")
    print("=" * 40)
    
    # Create sample RAG chain
    rag = create_sample_rag_chain("faiss")
    
    # Test queries
    test_questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the applications of deep learning?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        response = rag.query(question)
        print(f"🤖 Answer: {response.answer[:200]}...")
        print(f"📚 Documents retrieved: {len(response.retrieved_documents)}")
        print(f"🔧 Metadata: {response.metadata}")
    
    # Show chain info
    print(f"\n📊 Chain Info: {rag.get_chain_info()}")
