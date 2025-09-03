# RAG Assistant - Agentic AI Developer Certification Project

A Retrieval-Augmented Generation (RAG) powered conversational assistant that answers questions based on a custom knowledge base using LangChain and vector storage.

## 🎯 Project Overview

This project demonstrates a complete RAG pipeline that:
- Ingests documents into a vector store (FAISS/Chroma)
- Retrieves relevant context based on user queries
- Generates accurate responses using an LLM
- Provides both CLI and Streamlit web interface

## 🚀 Features

- **Document Ingestion**: Support for PDF, DOCX, and text files
- **Vector Storage**: FAISS and Chroma integration for efficient retrieval
- **Smart Retrieval**: Semantic search with configurable similarity thresholds
- **LLM Integration**: OpenAI GPT models for response generation
- **Multiple Interfaces**: CLI and Streamlit web UI
- **Session Memory**: Optional conversation memory for context retention
- **Logging & Observability**: Comprehensive logging for debugging

## 📁 Project Structure

```
rag-assistant/
├── src/
│   ├── __init__.py
│   ├── document_loader.py      # Document ingestion utilities
│   ├── vector_store.py         # Vector store management
│   ├── rag_chain.py           # Core RAG pipeline
│   ├── memory.py              # Conversation memory
│   └── utils.py               # Utility functions
├── data/                      # Sample documents and datasets
├── examples/                  # Example queries and outputs
├── cli.py                     # Command-line interface
├── app.py                     # Streamlit web application
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd rag-assistant
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

## 🔧 Configuration

Create a `.env` file with your configuration:

```env
OPENAI_API_KEY=your_openai_api_key_here
VECTOR_STORE_TYPE=faiss  # or chroma
MODEL_NAME=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=1000
```

## 📚 Usage

### 1. Document Ingestion

```python
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore

# Load documents
loader = DocumentLoader()
documents = loader.load_documents("data/")

# Create and populate vector store
vector_store = VectorStore(store_type="faiss")
vector_store.add_documents(documents)
```

### 2. RAG Query

```python
from src.rag_chain import RAGChain

# Initialize RAG chain
rag = RAGChain(vector_store)

# Ask a question
response = rag.query("What is this document about?")
print(response)
```

### 3. Command Line Interface

```bash
python cli.py --query "Your question here"
```

### 4. Web Interface

```bash
streamlit run app.py
```

## 🧪 Example Queries

Here are some sample questions to test your RAG assistant:

- "What is the main topic of this document?"
- "What are the key findings or conclusions?"
- "What methodology was used in this research?"
- "What are the limitations mentioned?"
- "Can you summarize the main points?"

## 🔍 Sample Datasets

The project includes sample datasets for testing:

- **Ready Tensor Publications**: Sample research papers and publications
- **Technical Documentation**: LangChain and related documentation
- **Custom Documents**: Your own document collection

## 🚀 Advanced Features

### Memory and Context
- Session-based conversation memory
- Context window management
- Multi-turn conversation support

### Enhanced Retrieval
- Hybrid search (semantic + keyword)
- Reranking for better relevance
- Configurable retrieval parameters

### Observability
- Query logging and analytics
- Performance metrics
- Error tracking and debugging

## 📊 Evaluation

The assistant is evaluated on:
- **Retrieval Quality**: Relevance of retrieved documents
- **Response Accuracy**: Correctness of generated answers
- **Response Relevance**: How well answers address the query
- **Performance**: Speed and efficiency of the pipeline

## 🤝 Contributing

This is a certification project, but feel free to:
- Report issues
- Suggest improvements
- Share your own implementations

## 📄 License

This project is created for educational purposes as part of the Agentic AI Developer Certification Program.

## 🙏 Acknowledgments

- LangChain team for the excellent framework
- OpenAI for LLM capabilities
- Ready Tensor for the certification program

---

**Built with ❤️ for the Agentic AI Developer Certification Program**
