# 🚀 RAG Assistant Quick Start Guide

Get your RAG Assistant up and running in minutes!

## ⚡ Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Set Up Environment
```bash
# Copy environment template
cp env.example .env

# Edit .env with your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Test the System
```bash
# Run comprehensive tests
python test_rag_assistant.py

# Or test individual components
python -c "from src.document_loader import create_sample_documents; print('✅ Document loader works!')"
```

## 🎯 Three Ways to Use

### Option 1: Command Line Interface (CLI)
```bash
# Interactive mode with sample data
python cli.py --interactive --sample

# Single question
python cli.py --query "What is artificial intelligence?" --sample

# Use your own documents
python cli.py --interactive --data-dir ./my_documents
```

### Option 2: Web Interface (Streamlit)
```bash
# Launch web app
streamlit run app.py

# Then open http://localhost:8501 in your browser
```

### Option 3: Python API
```python
from src.rag_chain import create_sample_rag_chain

# Create RAG chain
rag = create_sample_rag_chain("faiss")

# Ask questions
response = rag.query("What is machine learning?")
print(response.answer)
```

## 📚 Sample Data

The system includes sample AI/ML documents for testing:
- `data/sample_ai_document.txt` - AI overview
- `data/machine_learning_basics.txt` - ML fundamentals

## 🔧 Configuration Options

Key settings in `.env`:
- `VECTOR_STORE_TYPE`: Choose `faiss` (faster) or `chroma` (more features)
- `MODEL_NAME`: OpenAI model (gpt-3.5-turbo, gpt-4, etc.)
- `TOP_K`: Number of documents to retrieve (default: 5)
- `TEMPERATURE`: Response creativity (0.0 = focused, 1.0 = creative)

## 🧪 Testing Your Setup

### Basic Functionality Test
```bash
python -c "
from src.vector_store import create_sample_vector_store
store = create_sample_vector_store('faiss')
results = store.similarity_search('What is AI?', k=2)
print(f'✅ Found {len(results)} relevant documents')
"
```

### Full System Test
```bash
python test_rag_assistant.py
```

## 🚨 Common Issues & Solutions

### Issue: "No module named 'langchain'"
**Solution:** Install requirements: `pip install -r requirements.txt`

### Issue: "OPENAI_API_KEY not found"
**Solution:** Set your API key in `.env` file

### Issue: "Vector store creation failed"
**Solution:** Check if you have sample data or documents in the data directory

### Issue: "Memory error with large documents"
**Solution:** Reduce `chunk_size` in DocumentLoader or use smaller documents

## 📖 Next Steps

1. **Add Your Documents**: Place PDF, DOCX, or TXT files in the `data/` directory
2. **Customize Prompts**: Modify prompts in `src/rag_chain.py`
3. **Extend Functionality**: Add new document loaders or vector stores
4. **Deploy**: Use the Streamlit app for production deployment

## 🎉 You're Ready!

Your RAG Assistant is now ready to:
- Answer questions from your documents
- Maintain conversation context
- Provide source citations
- Handle multiple document formats

Start asking questions and exploring the power of RAG!
