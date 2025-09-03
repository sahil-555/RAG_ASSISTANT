"""
Conversation memory management for the RAG Assistant project.

This module provides conversation memory capabilities to maintain context
across multiple interactions with the RAG assistant.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    timestamp: datetime
    user_query: str
    assistant_response: str
    retrieved_documents: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """Manages conversation memory and context for multi-turn interactions."""
    
    def __init__(self, max_turns: int = 10, context_window: int = 5):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of turns to remember
            context_window: Number of recent turns to include in context
        """
        self.max_turns = max_turns
        self.context_window = context_window
        self.conversation_history: List[ConversationTurn] = []
        self.session_id = self._generate_session_id()
        
        logger.info(f"Initialized conversation memory with max_turns={max_turns}, context_window={context_window}")
    
    def add_interaction(self, user_query: str, assistant_response: str, retrieved_docs: int = 0, **metadata):
        """
        Add a new interaction to the conversation memory.
        
        Args:
            user_query: User's question or input
            assistant_response: Assistant's response
            retrieved_docs: Number of documents retrieved
            **metadata: Additional metadata for the interaction
        """
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_query=user_query,
            assistant_response=assistant_response,
            retrieved_documents=retrieved_docs,
            metadata=metadata
        )
        
        self.conversation_history.append(turn)
        
        # Maintain max_turns limit
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history.pop(0)
        
        logger.info(f"Added interaction to memory. Total turns: {len(self.conversation_history)}")
    
    def add_context(self, current_query: str) -> str:
        """
        Add conversation context to the current query.
        
        Args:
            current_query: Current user query
            
        Returns:
            Query with added context
        """
        if not self.conversation_history:
            return current_query
        
        # Get recent context
        recent_turns = self.conversation_history[-self.context_window:]
        
        context_parts = []
        for i, turn in enumerate(recent_turns, 1):
            context_parts.append(f"Turn {i}:")
            context_parts.append(f"User: {turn.user_query}")
            context_parts.append(f"Assistant: {turn.assistant_response}")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Combine with current query
        enhanced_query = f"""Previous conversation context:
{context}

Current question: {current_query}

Please consider the conversation history when answering the current question."""
        
        logger.info(f"Enhanced query with {len(recent_turns)} turns of context")
        return enhanced_query
    
    def get_recent_context(self, num_turns: Optional[int] = None) -> List[ConversationTurn]:
        """
        Get recent conversation context.
        
        Args:
            num_turns: Number of recent turns to return (defaults to context_window)
            
        Returns:
            List of recent conversation turns
        """
        if num_turns is None:
            num_turns = self.context_window
        
        return self.conversation_history[-num_turns:] if self.conversation_history else []
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        if not self.conversation_history:
            return {"message": "No conversation history"}
        
        total_turns = len(self.conversation_history)
        total_documents = sum(turn.retrieved_documents for turn in self.conversation_history)
        
        # Calculate average response length
        avg_response_length = sum(len(turn.assistant_response) for turn in self.conversation_history) / total_turns
        
        # Get conversation duration
        if total_turns > 1:
            duration = self.conversation_history[-1].timestamp - self.conversation_history[0].timestamp
            duration_str = str(duration)
        else:
            duration_str = "N/A"
        
        return {
            "session_id": self.session_id,
            "total_turns": total_turns,
            "total_documents_retrieved": total_documents,
            "average_response_length": round(avg_response_length, 2),
            "conversation_duration": duration_str,
            "start_time": self.conversation_history[0].timestamp.isoformat() if self.conversation_history else None,
            "last_interaction": self.conversation_history[-1].timestamp.isoformat() if self.conversation_history else None
        }
    
    def search_conversation(self, query: str) -> List[Tuple[ConversationTurn, float]]:
        """
        Search through conversation history for relevant interactions.
        
        Args:
            query: Search query
            
        Returns:
            List of tuples with (turn, relevance_score)
        """
        if not self.conversation_history:
            return []
        
        # Simple keyword-based search (could be enhanced with embeddings)
        query_lower = query.lower()
        results = []
        
        for turn in self.conversation_history:
            score = 0
            
            # Check user query
            if query_lower in turn.user_query.lower():
                score += 2
            
            # Check assistant response
            if query_lower in turn.assistant_response.lower():
                score += 1
            
            if score > 0:
                results.append((turn, score))
        
        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def clear_memory(self):
        """Clear all conversation memory."""
        self.conversation_history.clear()
        self.session_id = self._generate_session_id()
        logger.info("Conversation memory cleared")
    
    def export_memory(self, filepath: str):
        """
        Export conversation memory to a JSON file.
        
        Args:
            filepath: Path to save the exported memory
        """
        try:
            export_data = {
                "session_id": self.session_id,
                "max_turns": self.max_turns,
                "context_window": self.context_window,
                "conversation_history": [
                    {
                        "timestamp": turn.timestamp.isoformat(),
                        "user_query": turn.user_query,
                        "assistant_response": turn.assistant_response,
                        "retrieved_documents": turn.retrieved_documents,
                        "metadata": turn.metadata
                    }
                    for turn in self.conversation_history
                ]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Conversation memory exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting conversation memory: {e}")
            raise
    
    def import_memory(self, filepath: str):
        """
        Import conversation memory from a JSON file.
        
        Args:
            filepath: Path to the memory file to import
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Clear current memory
            self.clear_memory()
            
            # Import settings
            self.max_turns = import_data.get("max_turns", self.max_turns)
            self.context_window = import_data.get("context_window", self.context_window)
            self.session_id = import_data.get("session_id", self.session_id)
            
            # Import conversation history
            for turn_data in import_data.get("conversation_history", []):
                turn = ConversationTurn(
                    timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                    user_query=turn_data["user_query"],
                    assistant_response=turn_data["assistant_response"],
                    retrieved_documents=turn_data["retrieved_documents"],
                    metadata=turn_data.get("metadata", {})
                )
                self.conversation_history.append(turn)
            
            logger.info(f"Conversation memory imported from {filepath}")
            
        except Exception as e:
            logger.error(f"Error importing conversation memory: {e}")
            raise
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "session_id": self.session_id,
            "current_turns": len(self.conversation_history),
            "max_turns": self.max_turns,
            "context_window": self.context_window,
            "memory_usage_percent": (len(self.conversation_history) / self.max_turns) * 100
        }


class PersistentMemory(ConversationMemory):
    """Persistent conversation memory that saves to disk automatically."""
    
    def __init__(self, storage_path: str = "memory", **kwargs):
        """
        Initialize persistent memory.
        
        Args:
            storage_path: Directory to store memory files
            **kwargs: Additional arguments for ConversationMemory
        """
        super().__init__(**kwargs)
        self.storage_path = storage_path
        self.memory_file = f"{storage_path}/conversation_memory.json"
        
        # Create storage directory
        import os
        os.makedirs(storage_path, exist_ok=True)
        
        # Try to load existing memory
        self._load_memory()
        
        logger.info(f"Initialized persistent memory with storage at {storage_path}")
    
    def add_interaction(self, *args, **kwargs):
        """Add interaction and save to disk."""
        super().add_interaction(*args, **kwargs)
        self._save_memory()
    
    def clear_memory(self):
        """Clear memory and save to disk."""
        super().clear_memory()
        self._save_memory()
    
    def _save_memory(self):
        """Save memory to disk."""
        try:
            self.export_memory(self.memory_file)
        except Exception as e:
            logger.error(f"Error saving memory to disk: {e}")
    
    def _load_memory(self):
        """Load memory from disk."""
        import os
        if os.path.exists(self.memory_file):
            try:
                self.import_memory(self.memory_file)
                logger.info("Loaded existing conversation memory from disk")
            except Exception as e:
                logger.warning(f"Could not load existing memory: {e}")


if __name__ == "__main__":
    # Test the memory module
    print("🧠 Conversation Memory Test")
    print("=" * 40)
    
    # Create memory
    memory = ConversationMemory(max_turns=5, context_window=3)
    
    # Add some interactions
    interactions = [
        ("What is AI?", "AI is artificial intelligence...", 2),
        ("How does it work?", "AI works by processing data...", 3),
        ("What are the benefits?", "AI provides automation...", 1),
    ]
    
    for query, response, docs in interactions:
        memory.add_interaction(query, response, docs)
        print(f"Added: {query} -> {response[:30]}...")
    
    # Test context addition
    current_query = "Can you summarize our conversation?"
    enhanced_query = memory.add_context(current_query)
    print(f"\nEnhanced query:\n{enhanced_query[:200]}...")
    
    # Show summary
    summary = memory.get_conversation_summary()
    print(f"\nConversation summary: {summary}")
    
    # Test search
    search_results = memory.search_conversation("AI")
    print(f"\nSearch results for 'AI': {len(search_results)} found")
    
    # Show memory stats
    stats = memory.get_memory_stats()
    print(f"\nMemory stats: {stats}")
