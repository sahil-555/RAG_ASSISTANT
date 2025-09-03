"""
Utility functions for the RAG Assistant project.

This module provides common utility functions used across the project
including logging, text processing, and file operations.
"""

import os
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import re


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        log_format: Log message format
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("rag_assistant")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    console_handler.setFormatter(formatter)
    
    # Add console handler
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
    
    return text


def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using Jaccard similarity.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    # Convert to sets of words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of keywords
    """
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Clean and split text
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out stop words and short words
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]


def create_file_hash(file_path: str) -> str:
    """
    Create a hash of a file for integrity checking.
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA-256 hash of the file
    """
    hash_sha256 = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logging.error(f"Error creating file hash for {file_path}: {e}")
        return ""


def save_json(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
        indent: JSON indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
        
        return True
    except Exception as e:
        logging.error(f"Error saving JSON to {file_path}: {e}")
        return False


def load_json(file_path: str) -> Optional[Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading JSON from {file_path}: {e}")
        return None


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    try:
        path = Path(file_path)
        stat = path.stat()
        
        return {
            "name": path.name,
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": path.suffix,
            "hash": create_file_hash(file_path)
        }
    except Exception as e:
        logging.error(f"Error getting file info for {file_path}: {e}")
        return {}


def create_sample_data(data_dir: str = "data") -> None:
    """
    Create sample data files for testing.
    
    Args:
        data_dir: Directory to create sample data in
    """
    os.makedirs(data_dir, exist_ok=True)
    
    sample_texts = [
        {
            "filename": "ai_introduction.txt",
            "content": """
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines that work and react like humans. Some of the activities 
            computers with artificial intelligence are designed for include speech recognition, 
            learning, planning, and problem solving.
            
            AI has been around for decades, but recent advances in machine learning and 
            deep learning have made it more powerful and accessible than ever before.
            """
        },
        {
            "filename": "machine_learning_basics.txt",
            "content": """
            Machine Learning is a subset of AI that provides systems the ability to automatically 
            learn and improve from experience without being explicitly programmed. Machine learning 
            focuses on the development of computer programs that can access data and use it to 
            learn for themselves.
            
            There are three main types of machine learning: supervised learning, unsupervised 
            learning, and reinforcement learning.
            """
        },
        {
            "filename": "deep_learning_overview.txt",
            "content": """
            Deep Learning is a subset of machine learning that uses neural networks with multiple 
            layers to model and understand complex patterns. It's particularly effective for 
            tasks like image recognition, natural language processing, and speech recognition.
            
            Deep learning models require large amounts of data and computational power, but 
            they can achieve state-of-the-art results on many tasks.
            """
        }
    ]
    
    for sample in sample_texts:
        file_path = os.path.join(data_dir, sample["filename"])
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample["content"].strip())
        
        logging.info(f"Created sample file: {file_path}")


def validate_environment() -> Dict[str, bool]:
    """
    Validate the environment and dependencies.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {}
    
    # Check Python version
    import sys
    validation_results["python_version"] = sys.version_info >= (3, 8)
    
    # Check required packages
    required_packages = [
        "langchain", "openai", "faiss-cpu", "chromadb", 
        "sentence-transformers", "streamlit"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            validation_results[f"package_{package}"] = True
        except ImportError:
            validation_results[f"package_{package}"] = False
    
    # Check environment variables
    validation_results["openai_api_key"] = bool(os.getenv("OPENAI_API_KEY"))
    
    # Check directories
    validation_results["data_directory"] = os.path.exists("data")
    validation_results["vector_store_directory"] = os.path.exists("vector_store")
    
    return validation_results


def format_timestamp(timestamp: Union[str, datetime]) -> str:
    """
    Format timestamp in a human-readable format.
    
    Args:
        timestamp: Timestamp to format
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            return timestamp
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def create_progress_bar(iterable, description: str = "Processing"):
    """
    Create a progress bar for iterables.
    
    Args:
        iterable: Iterable to wrap
        description: Description for the progress bar
        
    Returns:
        Progress bar wrapper
    """
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=description)
    except ImportError:
        # Fallback without tqdm
        return iterable


if __name__ == "__main__":
    # Test utility functions
    print("🔧 Utility Functions Test")
    print("=" * 40)
    
    # Test text processing
    sample_text = "This is a sample text with some words and more words."
    print(f"Original text: {sample_text}")
    print(f"Cleaned text: {clean_text(sample_text)}")
    print(f"Truncated text: {truncate_text(sample_text, 20)}")
    print(f"Keywords: {extract_keywords(sample_text)}")
    
    # Test file operations
    test_data = {"test": "data", "number": 42}
    test_file = "test_output.json"
    
    if save_json(test_data, test_file):
        print(f"Saved JSON to {test_file}")
        loaded_data = load_json(test_file)
        print(f"Loaded data: {loaded_data}")
        
        # Clean up
        os.remove(test_file)
    
    # Test environment validation
    env_status = validate_environment()
    print(f"\nEnvironment validation: {env_status}")
    
    # Test sample data creation
    create_sample_data("test_data")
    print("Created sample data files")
    
    # Clean up
    import shutil
    shutil.rmtree("test_data", ignore_errors=True)
