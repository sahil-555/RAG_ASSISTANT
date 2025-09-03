"""
Sample Ready Tensor Publications Dataset

This file contains sample publications that mimic the Ready Tensor platform
content for testing the RAG assistant.
"""

SAMPLE_PUBLICATIONS = [
    {
        "title": "Advanced RAG Systems with Multi-Modal Retrieval",
        "content": """
        This publication explores the development of advanced Retrieval-Augmented Generation (RAG) systems that incorporate multi-modal retrieval capabilities. The research demonstrates how combining text, image, and structured data retrieval can significantly improve the quality of AI-generated responses.

        Key Findings:
        - Multi-modal retrieval improves response accuracy by 23%
        - Hybrid search strategies outperform single-modal approaches
        - Context-aware retrieval reduces hallucination by 15%

        Methodology:
        The study employed a transformer-based architecture with attention mechanisms for cross-modal alignment. We used a dataset of 10,000 multi-modal documents and evaluated performance using BLEU, ROUGE, and human assessment metrics.

        Limitations:
        - Requires significant computational resources
        - Limited to English language content
        - Training data may contain biases

        Tools and Models Used:
        - BERT for text encoding
        - Vision Transformer for image processing
        - Custom attention mechanism for cross-modal fusion
        """,
        "author": "Dr. Sarah Chen",
        "tags": ["RAG", "Multi-Modal", "AI", "Retrieval"]
    },
    {
        "title": "Efficient Vector Database Management for Large-Scale RAG",
        "content": """
        This paper presents novel approaches to managing vector databases for large-scale RAG applications. We introduce a hierarchical indexing structure that reduces search time by 40% while maintaining retrieval quality.

        Key Findings:
        - Hierarchical indexing improves search efficiency
        - Adaptive clustering reduces memory usage by 30%
        - Dynamic rebalancing maintains performance under load

        Methodology:
        We implemented a hierarchical HNSW (Hierarchical Navigable Small World) graph with adaptive clustering. The system was tested on a corpus of 1 million documents with real-time query processing.

        Limitations:
        - Initial setup complexity
        - Requires periodic rebalancing
        - Memory overhead for small datasets

        Tools and Models Used:
        - HNSW algorithm
        - K-means clustering
        - Custom rebalancing algorithm
        """,
        "author": "Prof. Michael Rodriguez",
        "tags": ["Vector Database", "HNSW", "Scalability", "Performance"]
    },
    {
        "title": "Conversational Memory in RAG Systems",
        "content": """
        This research investigates the role of conversational memory in RAG systems and its impact on user experience. We propose a novel memory architecture that maintains context across multiple conversation turns.

        Key Findings:
        - Contextual memory improves user satisfaction by 35%
        - Memory decay functions enhance conversation flow
        - Adaptive memory sizing optimizes performance

        Methodology:
        We developed a sliding window memory system with attention-based context selection. The system was evaluated on 500 multi-turn conversations with human evaluators.

        Limitations:
        - Memory size affects response time
        - Context selection can be computationally expensive
        - Requires careful tuning of decay parameters

        Tools and Models Used:
        - LSTM for sequence modeling
        - Attention mechanism for context selection
        - Custom decay functions
        """,
        "author": "Dr. Emily Watson",
        "tags": ["Memory", "Conversation", "Context", "User Experience"]
    },
    {
        "title": "Real-time RAG Systems for Enterprise Applications",
        "content": """
        This study examines the implementation of real-time RAG systems in enterprise environments, focusing on scalability, security, and integration challenges. We present a framework for deploying production-ready RAG systems.

        Key Findings:
        - Real-time processing reduces response latency by 60%
        - Enterprise-grade security features are essential
        - API integration simplifies deployment and maintenance

        Methodology:
        We developed a microservices-based architecture with real-time document indexing and secure API endpoints. The system was deployed in three enterprise environments with 1000+ concurrent users.

        Limitations:
        - Higher infrastructure costs
        - Requires dedicated DevOps resources
        - Complex monitoring and alerting setup

        Tools and Models Used:
        - Docker containers
        - Kubernetes orchestration
        - Redis for caching
        - JWT authentication
        """,
        "author": "Dr. James Wilson",
        "tags": ["Enterprise", "Real-time", "Scalability", "Security"]
    },
    {
        "title": "Evaluating RAG System Performance: A Comprehensive Framework",
        "content": """
        This publication introduces a comprehensive framework for evaluating RAG system performance across multiple dimensions. We propose standardized metrics and evaluation protocols for the RAG community.

        Key Findings:
        - Multi-dimensional evaluation provides better insights
        - Human evaluation remains crucial for quality assessment
        - Automated metrics correlate well with human judgment

        Methodology:
        We developed evaluation metrics covering retrieval accuracy, response quality, relevance, and user satisfaction. The framework was tested on 50 different RAG implementations.

        Limitations:
        - Evaluation can be time-consuming
        - Some metrics require human annotators
        - Results may vary across domains

        Tools and Models Used:
        - BLEU and ROUGE metrics
        - Custom relevance scoring
        - Human evaluation platform
        """,
        "author": "Dr. Lisa Thompson",
        "tags": ["Evaluation", "Metrics", "Performance", "Framework"]
    }
]

def create_publication_files():
    """Create sample publication files for testing."""
    import os
    
    # Create publications directory
    os.makedirs("data/publications", exist_ok=True)
    
    for i, pub in enumerate(SAMPLE_PUBLICATIONS):
        filename = f"data/publications/publication_{i+1}.txt"
        content = f"Title: {pub['title']}\nAuthor: {pub['author']}\nTags: {', '.join(pub['tags'])}\n\n{pub['content']}"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Created: {filename}")

if __name__ == "__main__":
    create_publication_files()
