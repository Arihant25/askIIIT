# Jagruti Backend

A FastAPI-based backend with ChromaDB for document search and chatbot functionality using **Qwen3 models** with CAS authentication.

## Features

- **FastAPI** REST API with automatic documentation
- **ChromaDB** for vector embeddings and semantic search
- **Qwen3-Embedding-8B** via sentence-transformers for high-quality embeddings
- **Qwen3 (8B)** via Ollama for chat and summarization
- **CAS Authentication** integration for IIIT login
- **Document Processing** with text extraction and chunking
- **Semantic Search** using Qwen3 embeddings
- **Admin Panel** endpoints for document management
- **Bulk Processing** for existing PDF documents

## Models Used

- **Embeddings**: `Qwen/Qwen3-Embedding-8B` via Hugging Face sentence-transformers
- **Chat/Summarization**: `qwen3:8b` via Ollama

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install and Setup Ollama

Download and install Ollama from [https://ollama.ai/download](https://ollama.ai/download)

Start Ollama server:

```bash
ollama serve
```

Pull the Qwen model:

```bash
ollama pull qwen3:8b
```

### 3. Run Setup Script

The setup script will install dependencies, setup models, and test the system:

```bash
python setup.py
```

### 4. Environment Configuration

The `.env` file is pre-configured with defaults. Update these settings as needed:

```env
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen3:8b

# Embedding model (Hugging Face)
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
EMBEDDING_DEVICE=auto
EMBEDDING_TRUST_REMOTE_CODE=true

# CAS Authentication
CAS_SERVER_URL=https://login.iiit.ac.in/cas
ADMIN_USERS=admin@iiit.ac.in,arihant.tripathy@research.iiit.ac.in,mohit.singh@research.iiit.ac.in,aviral.gupta@research.iiit.ac.in
```

### 5. Start the Server

```bash
python run.py
```

Or directly with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Process Existing PDFs

Check existing documents:

```bash
python bulk_process.py --check
```

Process all PDFs in the pdfs directory:

```bash
python bulk_process.py --process
```

## API Endpoints

### Health & Status

- `GET /` - Basic health check
- `GET /health` - Comprehensive health check including model status

### Authentication

- `GET /auth/login` - Get CAS login URL
- `GET /auth/user` - Get current user info

### Documents

- `POST /api/documents/upload` - Upload new document (Admin only)
- `GET /api/documents` - List documents with filtering
- `GET /api/categories` - Get available categories
- `GET /api/stats` - Get system statistics

### Search & Chat

- `POST /api/search` - Semantic search using Qwen3 embeddings
- `POST /api/chat` - Chat with document context using Qwen3

## Model Architecture

### Embedding Pipeline

1. **Text Extraction**: PyPDF2 for PDFs, direct reading for text files
2. **Text Chunking**: Split into 500-character chunks with 50-character overlap
3. **Embedding Generation**: Qwen3-Embedding-8B via sentence-transformers
4. **Storage**: Vector embeddings stored in ChromaDB with metadata

### Chat Pipeline

1. **Query Processing**: Convert user query to embedding using Qwen3-Embedding-8B
2. **Similarity Search**: Find relevant chunks in ChromaDB
3. **Context Preparation**: Format relevant chunks as context
4. **Response Generation**: Qwen3 (8B) via Ollama generates response with context
5. **Response**: Natural language answer with source references

## API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## System Requirements

### Hardware

- **RAM**: 16GB+ recommended (8GB minimum)
- **GPU**: Optional but recommended for faster embedding generation
- **Storage**: 10GB+ for models and document storage

### Software

- **Python**: 3.8+
- **Ollama**: Latest version
- **CUDA**: Optional, for GPU acceleration

## Model Performance

### Qwen3-Embedding-8B

- **Dimension**: 1024
- **Performance**: State-of-the-art multilingual embeddings
- **Languages**: Excellent support for English and Chinese
- **Use Case**: Document similarity and semantic search

### Qwen3 (8B) Chat Model

- **Parameters**: 8 billion
- **Performance**: High-quality conversational AI
- **Context Length**: 32K tokens
- **Use Case**: Document-based Q&A and summarization

## File Structure

```
backend/
├── main.py                 # Main FastAPI application
├── document_processor.py   # Document processing with Qwen3 embeddings
├── ollama_client.py       # Ollama client for Qwen3 chat
├── auth_utils.py          # CAS authentication utilities
├── scraper.py             # Web scraping utilities
├── bulk_process.py        # Bulk PDF processing script
├── setup.py               # Setup and installation script
├── run.py                 # Startup script
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
└── chroma_data/          # ChromaDB storage (created automatically)
```

## Development

### Testing Models

Test the embedding model:

```bash
python -c "
from document_processor import DocumentProcessor
dp = DocumentProcessor()
emb = dp.generate_embeddings(['test'])
print(f'Embedding dimension: {len(emb[0])}')
"
```

Test the chat model:

```bash
python -c "
import asyncio
from ollama_client import ollama_client
result = asyncio.run(ollama_client.generate_response('Hello, how are you?'))
print(result)
"
```

### Performance Tuning

1. **GPU Acceleration**: Set `EMBEDDING_DEVICE=cuda` if you have a compatible GPU
2. **Batch Size**: Adjust batch size in `generate_embeddings()` based on your hardware
3. **Chunk Size**: Modify `CHUNK_SIZE` and `CHUNK_OVERLAP` for your use case
4. **Model Precision**: Use quantized models for lower memory usage

## Troubleshooting

### Common Issues

1. **Ollama connection fails**:

   ```bash
   ollama serve  # Start Ollama server
   ollama pull qwen3:8b  # Ensure model is downloaded
   ```

2. **Embedding model download fails**:

   - Check internet connection
   - Verify Hugging Face access
   - Try `trust_remote_code=true` in config

3. **Out of memory errors**:

   - Reduce batch size in embedding generation
   - Use CPU instead of GPU: `EMBEDDING_DEVICE=cpu`
   - Process documents in smaller batches

4. **Slow performance**:
   - Use GPU acceleration if available
   - Increase batch sizes
   - Consider using quantized models

### Logging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Production Deployment

1. **Security**:

   - Set `DEBUG=False`
   - Use proper session storage (Redis/database)
   - Configure proper CORS origins
   - Set up SSL/TLS certificates

2. **Performance**:

   - Use Gunicorn with multiple workers
   - Set up load balancing
   - Use GPU acceleration
   - Configure model caching

3. **Monitoring**:
   - Monitor model performance and response times
   - Set up health checks for Ollama
   - Track embedding generation metrics
   - Monitor ChromaDB storage usage

## Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Include docstrings for new functions
4. Test endpoints using the Swagger UI
5. Test model integrations thoroughly
6. Update this README for new features
