# Jagruti Backend

A FastAPI-based backend with ChromaDB for document search and chatbot functionality using **Qwen 3-0.6B** with CAS authentication.

## Features

- **FastAPI** REST API with automatic documentation
- **ChromaDB** for vector embeddings and semantic search
- **Qwen3-Embedding-0.6B** via sentence-transformers for embeddings
- **Qwen 3-0.6B** via Hugging Face Text Generation Inference for chat and summarization
- **CAS Authentication** integration for IIIT login
- **Document Processing** with text extraction and chunking
- **Semantic Search** using Qwen3 embeddings
- **Admin Panel** endpoints for document management
- **Bulk Processing** for existing PDF documents

## Models Used

- **Embeddings**: `Qwen/Qwen3-Embedding-8B` via Hugging Face sentence-transformers
- **Chat/Summarization**: `google/gemma-3-270m` via Hugging Face Text Generation Inference

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Setup Hugging Face Text Generation Inference

Start the HF TGI Docker container with Qwen model:

```bash
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id google/gemma-3-270m
```

### 3. Run Setup Script

The setup script will install dependencies, setup models, and test the system:

```bash
python setup.py
```

### 4. Environment Configuration

The `.env` file is pre-configured with defaults. Update these settings as needed:

```env
# HF TGI Configuration
OLLAMA_BASE_URL=http://localhost:8000
OLLAMA_CHAT_MODEL=google/gemma-3-270m

# Embedding model (Hugging Face)
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
EMBEDDING_DEVICE=auto
EMBEDDING_TRUST_REMOTE_CODE=true

# CAS Authentication
CAS_SERVER_URL=https://login.iiit.ac.in/cas
ADMIN_USERS=admin@iiit.ac.in,arihant.tripathy@research.iiit.ac.in,mohit.singh@research.iiit.ac.in,aviral.gupta@research.iiit.ac.in

# Search Configuration
RELEVANCE_THRESHOLD=0.7  # Distance threshold for search result filtering (0.0-1.0, lower = more strict)
```

### Search Relevance Configuration

The system now includes smart filtering to ensure only relevant documents are shown as references:

- **RELEVANCE_THRESHOLD**: Controls how strict the search filtering is
  - `0.5`: Very strict - only highly relevant content (fewer results)
  - `0.7`: Default - good balance of relevance and coverage
  - `0.9`: Lenient - includes more potentially relevant content

Use the debug endpoint to test different threshold values:

```bash
curl "http://localhost:8000/api/debug/search-relevance?query=your-test-query&threshold=0.6"
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
- `POST /api/chat` - Chat with document context using Qwen3 (with conversation history)
- `POST /api/chat/stream` - Streaming chat with real-time responses (with conversation history)
- `GET /api/debug/search-relevance` - Debug search relevance filtering (requires authentication)
- `POST /api/debug/test-conversation` - Debug conversation history handling

## Model Architecture

### Embedding Pipeline

1. **Text Extraction**: pdfplumber for PDFs, direct reading for text files
2. **Text Chunking**: Split into 400-character chunks with 50-character overlap (memory optimized)
3. **Embedding Generation**: Qwen3-Embedding-0.6B via sentence-transformers
4. **Storage**: Vector embeddings stored in ChromaDB with metadata

### Chat Pipeline

1. **Query Processing**: Convert user query to embedding using Qwen3-Embedding-0.6B
2. **Conversation Context**: Include previous messages from the conversation for continuity
3. **Similarity Search**: Find relevant chunks in ChromaDB
4. **Context Preparation**: Format relevant chunks and conversation history as context
5. **Response Generation**: Qwen 3-0.6B via Hugging Face TGI generates response with full context
6. **Response**: Natural language answer with source references and conversation memory

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
- **Docker**: For running Hugging Face Text Generation Inference
- **CUDA**: Optional, for GPU acceleration (highly recommended for Qwen model)

## Model Performance

### Qwen3-Embedding-0.6B

- **Dimension**: 384
- **Performance**: Lightweight yet effective embeddings
- **Languages**: Good support for English
- **Use Case**: Document similarity and semantic search with lower memory footprint

### Qwen 3-1.7B Chat Model

- **Parameters**: 0.6 billion
- **Performance**: Fast and efficient conversational AI
- **Context Length**: 8K tokens
- **Use Case**: Document-based Q&A and summarization
- **Deployment**: Via Hugging Face Text Generation Inference Docker container

## File Structure

```
backend/
├── main.py                 # Main FastAPI application
├── document_processor.py   # Document processing with Qwen3 embeddings
├── ollama_client.py        # Client for HF TGI inference model
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

1. **Hugging Face TGI server fails**:

   ```bash
   # Start HF TGI Docker container for Qwen
   docker run --gpus all \
       -v ~/.cache/huggingface:/root/.cache/huggingface \
       -p 8000:80 \
       ghcr.io/huggingface/text-generation-inference:latest \
       --model-id google/gemma-3-270m
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
   - Set up health checks for HF TGI
   - Track embedding generation metrics
   - Monitor ChromaDB storage usage

## Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Include docstrings for new functions
4. Test endpoints using the Swagger UI
5. Test model integrations thoroughly
6. Update this README for new features
