"""
Local LLM implementation using Hugging Face models.

This module provides a local alternative to OpenAI models for the RAG assistant.
"""

import logging
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

logger = logging.getLogger(__name__)


class LocalLLM:
    """Local language model using Hugging Face transformers."""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        max_length: int = 1000,
        temperature: float = 0.7,
        device: Optional[str] = None
    ):
        """
        Initialize the local LLM.
        
        Args:
            model_name: Hugging Face model name
            max_length: Maximum output length
            temperature: Sampling temperature
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        logger.info(f"Initializing local LLM: {model_name} on {device}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device if device != "cpu" else None
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device if device != "cpu" else -1,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info(f"Successfully loaded local LLM: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading local LLM: {e}")
            raise
    
    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Generated text
        """
        try:
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Generate response
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and return generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output
            response = generated_text[len(prompt):].strip()
            
            logger.info(f"Generated {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 200) -> str:
        """
        Generate a chat completion from messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
        """
        try:
            # Format messages into a prompt
            prompt = self._format_messages_to_prompt(messages)
            
            # Generate response
            response = self.generate(prompt, max_new_tokens=max_tokens)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into a single prompt string.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        # Add the final prompt for the assistant
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "model_type": "local_huggingface"
        }


# Alternative: Simple rule-based responses for testing
class SimpleRuleBasedLLM:
    """Simple rule-based LLM for testing when models aren't available."""
    
    def __init__(self):
        self.model_type = "rule_based"
        self.responses = {
            "artificial intelligence": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans.",
            "machine learning": "Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience.",
            "deep learning": "Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns.",
            "what is": "Based on the context provided, I can help answer your question. Please provide more specific information about what you'd like to know.",
            "how does": "I'd be happy to explain how this works based on the context. Could you please provide more details about what you're asking?",
            "explain": "I can explain this topic based on the information in the documents. What specific aspect would you like me to clarify?"
        }
    
    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate a simple rule-based response."""
        prompt_lower = prompt.lower()
        
        for key, response in self.responses.items():
            if key in prompt_lower:
                return response
        
        return "I understand your question. Based on the context provided, I can help you with information about AI, machine learning, and related topics. Could you please be more specific about what you'd like to know?"
    
    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 200) -> str:
        """Generate a chat completion response."""
        if not messages:
            return "Hello! How can I help you today?"
        
        # Get the last user message
        last_user_message = None
        for message in reversed(messages):
            if message.get('role') == 'user':
                last_user_message = message.get('content', '')
                break
        
        if last_user_message:
            return self.generate(last_user_message, max_new_tokens)
        
        return "I'm here to help! What would you like to know?"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about this simple model."""
        return {
            "model_name": "simple_rule_based",
            "device": "cpu",
            "max_length": 200,
            "temperature": 0.7,
            "model_type": "rule_based"
        }
