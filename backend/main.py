from colored_logging import setup_logging
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse, StreamingResponse
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
from dotenv import load_dotenv
import torch
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import uuid
import httpx
from urllib.parse import urlencode, quote
import logging
import json
import uvicorn

# Load environment variables
load_dotenv()

# Configure colored logging

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Jagruti API",
    description="Backend API for IIIT document search and chatbot",
    version="1.0.0",
)

# CORS middleware
cors_origins = os.getenv("CORS_ORIGINS", "").split(",")
if not cors_origins or cors_origins == [""]:
    # Default origins if not specified
    cors_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://10.2.58.213:3000",
    ]
# Allow all origins in development mode
if os.getenv("DEBUG", "True").lower() == "true":
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize ChromaDB client
try:
    chroma_client = chromadb.PersistentClient(
        path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
    )
    logger.info("ChromaDB client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    raise


# Pydantic models
class DocumentCategory(str):
    FACULTY = "faculty"
    STUDENT = "student"
    HOSTEL = "hostel"
    ACADEMICS = "academics"
    MESS = "mess"


class DocumentMetadata(BaseModel):
    doc_id: str
    name: str
    category: str
    description: str
    created_at: datetime
    updated_at: datetime
    file_url: str
    author: Optional[str] = None


class ChunkData(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_text: str
    position: int
    category: str


class QueryRequest(BaseModel):
    query: str
    categories: Optional[List[str]] = None
    limit: Optional[int] = 10


class ChatMessage(BaseModel):
    type: str  # 'user' or 'bot'
    content: str


class ChatRequest(BaseModel):
    message: str
    categories: Optional[List[str]] = None
    conversation_id: Optional[str] = None
    conversation_history: Optional[List[ChatMessage]] = None


class UserInfo(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    is_admin: bool = False


# CAS Authentication functions
async def validate_cas_token(ticket: str, service_url: str) -> Optional[UserInfo]:
    """Validate CAS ticket and return user info"""
    cas_server_url = os.getenv("CAS_SERVER_URL")
    validation_url = f"{cas_server_url}/serviceValidate"

    params = {"ticket": ticket, "service": service_url}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(validation_url, params=params)
            response.raise_for_status()

            # Parse CAS response (simplified - in production, parse XML properly)
            if "cas:authenticationSuccess" in response.text:
                # Extract username from CAS response
                # This is a simplified extraction - you should use proper XML parsing
                username = response.text.split("<cas:user>")[
                    1].split("</cas:user>")[0]

                # Check if user is admin
                admin_users = os.getenv("ADMIN_USERS", "").split(",")
                is_admin = f"{username}@iiit.ac.in" in admin_users

                return UserInfo(
                    username=username,
                    email=f"{username}@iiit.ac.in",
                    full_name=username.title(),
                    is_admin=is_admin,
                )
    except Exception as e:
        logger.error(f"CAS validation error: {e}")

    return None


async def get_current_user(request: Request) -> UserInfo:
    """Get current authenticated user from session or CAS ticket"""
    # Check for CAS ticket in query parameters
    ticket = request.query_params.get("ticket")
    if ticket:
        service_url = str(request.url).split("?")[0]  # Remove query parameters
        user = await validate_cas_token(ticket, service_url)
        if user:
            return user

    # Check for existing session (you can implement session storage here)
    # For now, we'll check for Authorization header or cookie
    authorization = request.headers.get("Authorization")
    if authorization and authorization.startswith("Bearer "):
        # In a real implementation, you'd validate the JWT token here
        # For now, we'll return a mock user for development
        return UserInfo(
            username="testuser",
            email="testuser@iiit.ac.in",
            full_name="Test User",
            is_admin=True,
        )

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Please login through CAS.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_admin_user(
    current_user: UserInfo = Depends(get_current_user),
) -> UserInfo:
    """Ensure current user is an admin"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=403, detail="Admin privileges required")
    return current_user


# Custom embedding function for ChromaDB
try:
    # Check environment variables for device selection
    force_cpu_env = os.getenv("EMBEDDING_FORCE_CPU", "0").lower() in {"1", "true", "yes"}
    embedding_device_env = os.getenv("EMBEDDING_DEVICE", "auto").lower()
    
    # Determine device based on environment variables and availability
    if force_cpu_env:
        device = "cpu"
        logger.info("Forcing CPU usage for embeddings (EMBEDDING_FORCE_CPU=1)")
    elif embedding_device_env == "cpu":
        device = "cpu"
        logger.info("Using CPU for embeddings (EMBEDDING_DEVICE=cpu)")
    elif embedding_device_env == "cuda" and torch.cuda.is_available():
        device = "cuda"
        logger.info("Using CUDA for embeddings (EMBEDDING_DEVICE=cuda)")
    elif embedding_device_env == "mps" and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS for embeddings (EMBEDDING_DEVICE=mps)")
    elif embedding_device_env == "auto":
        # Auto-selection logic
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("Auto-selected MPS for embeddings")
        elif torch.cuda.is_available():
            # Check GPU memory for safety
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 8:  # Require at least 8GB GPU memory
                device = "cuda"
                logger.info(f"Auto-selected CUDA for embeddings ({gpu_memory_gb:.1f}GB GPU)")
            else:
                device = "cpu"
                logger.warning(f"GPU memory too low ({gpu_memory_gb:.1f}GB), auto-falling back to CPU")
        else:
            device = "cpu"
            logger.info("Auto-selected CPU for embeddings (no GPU available)")
    else:
        # Fallback to original logic
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using fallback device selection: {device}")

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device=device,
        normalize_embeddings=True
    )
    logger.info("Custom embedding function initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize embedding function: {e}")
    embedding_function = None

# Initialize ChromaDB collections
try:
    documents_collection = chroma_client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_function,
        metadata={"description": "Document metadata collection",
                  "hnsw:space": "cosine"}
    )
    chunks_collection = chroma_client.get_or_create_collection(
        name="chunks",
        embedding_function=embedding_function,
        metadata={"description": "Document chunks with embeddings",
                  "hnsw:space": "cosine"}
    )
    logger.info("ChromaDB collections initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize collections: {e}")
    raise





async def search_relevant_documents(message: str, categories: Optional[List[str]] = None, threshold: float = 0.5):
    """Search for relevant documents and return context chunks and metadata"""
    # Build where clause for filtering
    where_clause = {}
    if categories:
        if len(categories) == 1:
            where_clause["category"] = categories[0]
        else:
            where_clause["category"] = {"$in": categories}

    search_results = chunks_collection.query(
        query_texts=[message],
        n_results=10,  # Get more results to filter by relevance
        where=where_clause if where_clause else None,
    )

    # Filter results by relevance score (distance threshold)
    # Lower distance = higher similarity
    relevance_threshold = float(
        os.getenv("RELEVANCE_THRESHOLD", str(threshold)))

    context_chunks = []
    context_metadata = []

    if search_results["documents"] and search_results["distances"]:
        logger.info(
            f"Found {len(search_results['documents'][0])} potential chunks for query: '{message}' (threshold: {relevance_threshold})")
        
        logger.info("=" * 80)
        logger.info("ALL RETRIEVED CHUNKS:")
        logger.info("=" * 80)

        for i, (doc, metadata, distance) in enumerate(zip(
            search_results["documents"][0],
            search_results["metadatas"][0],
            search_results["distances"][0]
        )):
            relevance_score = 1.0 - distance
            logger.info(f"\n--- Chunk {i+1} ---")
            logger.info(f"Distance: {distance:.4f} | Relevance Score: {relevance_score:.4f}")
            logger.info(f"Filename: {metadata.get('filename', 'Unknown')}")
            logger.info(f"Category: {metadata.get('category', 'Unknown')}")
            logger.info(f"Doc ID: {metadata.get('doc_id', 'Unknown')}")
            logger.info(f"Chunk Position: {metadata.get('position', 'Unknown')}")
            logger.info(f"Text Preview: {doc[:300]}..." if len(doc) > 300 else f"Text: {doc}")
            logger.info(f"Full Text Length: {len(doc)} characters")
            
            # Only include chunks that are sufficiently relevant
            if distance <= relevance_threshold:
                context_chunks.append(doc)
                context_metadata.append(metadata)
                logger.info(f"✓ INCLUDED (distance={distance:.3f} <= threshold={relevance_threshold})")
            else:
                logger.info(f"✗ EXCLUDED (distance={distance:.3f} > threshold={relevance_threshold})")

            # Limit to top 5 relevant results
            if len(context_chunks) >= 5:
                logger.info(f"\n⚠️ Reached limit of 5 chunks, stopping further inclusion")
                break

        logger.info("=" * 80)
        logger.info(
            f"SUMMARY: Selected {len(context_chunks)} relevant chunks out of {len(search_results['documents'][0])} potential matches")
        logger.info("=" * 80)
    else:
        logger.warning(
            f"No search results found for query: '{message}'")

    # Prepare context information for frontend with actual relevance scores
    context_info = []
    search_distances = search_results.get("distances", [[]])[
        0] if search_results.get("distances") else []

    for i, (chunk, metadata) in enumerate(zip(context_chunks, context_metadata)):
        # Calculate relevance score (1 - distance) for display
        relevance_score = 1.0 - \
            search_distances[i] if i < len(search_distances) else 1.0

        context_info.append(
            {
                "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                "filename": metadata.get("filename", "Unknown"),
                "category": metadata.get("category", "Unknown"),
                "doc_id": metadata.get("doc_id", ""),
                "relevance_score": round(relevance_score, 3),
            }
        )

    return context_chunks, context_info


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Jagruti API is running!",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check including model availability"""
    try:
        from ollama_client import ollama_client
        from document_processor import DocumentProcessor

        health_status = {
            "api": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {},
        }

        # Check ChromaDB
        try:
            doc_count = documents_collection.count()
            chunk_count = chunks_collection.count()
            health_status["services"]["chromadb"] = {
                "status": "healthy",
                "documents": doc_count,
                "chunks": chunk_count,
            }
        except Exception as e:
            health_status["services"]["chromadb"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Check HF TGI connection
        try:
            hf_tgi_available = await ollama_client.check_connection()
            if hf_tgi_available:
                models = await ollama_client.list_models()
                health_status["services"]["hf_tgi"] = {
                    "status": "healthy",
                    "available_models": [model.get("id", model.get("name", "unknown")) for model in models],
                    "chat_model": ollama_client.chat_model,
                }
            else:
                health_status["services"]["hf_tgi"] = {
                    "status": "unhealthy",
                    "error": "Cannot connect to HF TGI server",
                }
        except Exception as e:
            health_status["services"]["hf_tgi"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Check embedding model
        try:
            doc_processor = DocumentProcessor()
            # Test with a small embedding
            test_embeddings = doc_processor.generate_embeddings(["test"])
            health_status["services"]["embeddings"] = {
                "status": "healthy",
                "model": os.getenv(
                    "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
                ),
                "embedding_dimension": (
                    len(test_embeddings[0]) if test_embeddings else 0
                ),
            }
        except Exception as e:
            health_status["services"]["embeddings"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Overall status
        all_healthy = all(
            service.get("status") == "healthy"
            for service in health_status["services"].values()
        )
        health_status["status"] = "healthy" if all_healthy else "degraded"

        return health_status

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@app.get("/auth/login")
async def login_redirect():
    """Redirect to CAS login"""
    cas_server_url = os.getenv("CAS_SERVER_URL")
    service_url = os.getenv("CAS_SERVICE_URL")

    login_url = f"{cas_server_url}/login?service={quote(service_url)}"
    return {"login_url": login_url}


@app.get("/auth/user")
async def get_user_info(current_user: UserInfo = Depends(get_current_user)):
    """Get current user information"""
    return current_user


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Form(...),
    description: Optional[str] = Form(None),
    current_user: UserInfo = Depends(get_admin_user),
):
    """Upload a new document (Admin only)"""
    try:
        # Import document processor here to avoid circular imports
        from document_processor import DocumentProcessor, DocumentSummarizer

        # Validate file type
        allowed_extensions = os.getenv(
            "ALLOWED_EXTENSIONS", ".pdf,.txt,.doc,.docx"
        ).split(",")
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not allowed. Allowed types: {allowed_extensions}",
            )

        # Validate file size
        max_size = int(os.getenv("MAX_FILE_SIZE", 50000000))  # 50MB default
        file_content = await file.read()
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of {max_size} bytes",
            )

        # Create document ID
        doc_id = str(uuid.uuid4())

        # Initialize processors
        doc_processor = DocumentProcessor()
        summarizer = DocumentSummarizer()

        # Auto-generate description if not provided
        if not description:
            try:
                # Extract a small sample of text for summary
                sample_text = doc_processor.extract_text_from_file(
                    file_content, file.filename
                )
                description = summarizer.generate_summary(sample_text[:2000])
            except Exception as e:
                logger.warning(f"Could not generate description: {e}")
                description = f"Document: {file.filename}"

        # Process the document
        result = doc_processor.process_document(
            file_content=file_content,
            filename=file.filename,
            doc_id=doc_id,
            category=category,
            description=description,
            author=current_user.username,
            documents_collection=documents_collection,
            chunks_collection=chunks_collection,
        )

        logger.info(
            f"Document {file.filename} processed successfully: {result}")

        return {
            "message": "Document uploaded and processed successfully",
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_count": result["chunk_count"],
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process document: {str(e)}"
        )


@app.get("/api/documents")
async def list_documents(
    categories: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: UserInfo = Depends(get_current_user),
):
    """List documents with optional filtering"""
    try:
        # Query documents collection
        where_clause = {}
        if categories:
            # Support both single category (backward compatibility) and comma-separated categories
            category_list = [cat.strip() for cat in categories.split(",")]
            if len(category_list) == 1:
                where_clause["category"] = category_list[0]
            else:
                where_clause["category"] = {"$in": category_list}

        results = documents_collection.get(
            where=where_clause if where_clause else None, limit=limit, offset=offset
        )

        return {
            "documents": results["metadatas"],
            "total": len(results["ids"]),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve documents")


@app.post("/api/search")
async def search_documents(
    query_request: QueryRequest, current_user: UserInfo = Depends(get_current_user)
):
    """Search documents using semantic search"""
    try:
        # Build where clause for filtering
        where_clause = {}
        if query_request.categories:
            if len(query_request.categories) == 1:
                where_clause["category"] = query_request.categories[0]
            else:
                where_clause["category"] = {"$in": query_request.categories}

        # Perform semantic search on chunks with relevance filtering
        results = chunks_collection.query(
            query_texts=[query_request.query],
            n_results=query_request.limit or 10,
            where=where_clause if where_clause else None,
        )

        # Filter and format results with relevance scores
        formatted_results = []
        if results["documents"] and results["distances"]:
            for chunk_id, document, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # Only include results with reasonable relevance
                if distance <= 0.8:  # Adjust threshold as needed
                    formatted_results.append({
                        "chunk_id": chunk_id,
                        "text": document,
                        "metadata": metadata,
                        "distance": distance,
                        "relevance_score": round(1.0 - distance, 3),
                    })

        return {
            "query": query_request.query,
            "results": formatted_results,
            "total_found": len(formatted_results),
        }
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/api/chat")
async def chat_with_documents(chat_request: ChatRequest, request: Request):
    """Chat interface with document context using Qwen3 - No authentication required"""
    try:
        from ollama_client import ollama_client

        logger.info(f"Processing chat request")

        # Search for relevant documents
        context_chunks, context_info = await search_relevant_documents(
            chat_request.message,
            chat_request.categories,
            threshold=0.5
        )
        
        # Extract context metadata for tracking
        context_metadata = context_info

        # Prepare system prompt for Qwen3
        base_system_prompt = (
            f"You are Jagruti, a helpful assistant for IIIT Hyderabad. "
            f"You help students, faculty, and staff find information from official documents. "
            f"Use the provided context to answer questions accurately. "
            f"If you cannot find relevant information in the context, say so politely. "
            f"DO NOT answer questions that are not related to IIIT Hyderabad or if the information is not available in provided context. "
            f"When answering, naturally mention the document names you are referencing (e.g., 'According to [document name]...'). "
            f"Always be helpful, concise, and reference the source documents when applicable. "
            f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."
        )

        # Build conversation context from client-provided history
        conversation_context = ""
        if chat_request.conversation_history and len(chat_request.conversation_history) > 0:
            conversation_context = "\n\nPrevious conversation:\n"
            # Use last 10 messages for context
            for msg in chat_request.conversation_history[-10:]:
                role = "Human" if msg.type == "user" else "Assistant"
                conversation_context += f"{role}: {msg.content}\n"
            conversation_context += "Please maintain context from this conversation."

        # Build full system prompt with conversation history
        full_system_prompt = base_system_prompt + conversation_context

        # Generate response
        response = await ollama_client.generate_response(
            prompt=chat_request.message,
            context=context_chunks,
            system_prompt=full_system_prompt,
            context_metadata=context_metadata
        )

        # Extract which files were actually referenced in the response
        available_files = [meta.get('filename', 'Unknown') for meta in context_metadata]
        referenced_files = ollama_client.extract_referenced_files(response, available_files)
        
        # Filter context_info to only include referenced files
        filtered_context_info = [
            info for info in context_info 
            if info.get('filename') in referenced_files
        ]
        
        logger.info(f"Original context: {len(context_info)} files, Referenced: {len(filtered_context_info)} files")

        return {
            "message": chat_request.message,
            "response": response,
            "context_chunks": filtered_context_info,
            "context_found": len(context_chunks) > 0,
            "conversation_id": chat_request.conversation_id or str(uuid.uuid4()),
            "model_used": ollama_client.chat_model,
        }

    except Exception as e:
        logger.error(f"Error in chat: {e}")

        # Fallback response
        fallback_response = (
            "I apologize, but I'm having trouble processing your request right now. "
            "This might be because the Qwen model is not available or there's a connection issue. "
            "Please try again later or contact support if the problem persists."
        )

        return {
            "message": chat_request.message,
            "response": fallback_response,
            "context_chunks": [],
            "context_found": False,
            "conversation_id": chat_request.conversation_id or str(uuid.uuid4()),
            "error": str(e),
        }


@app.post("/api/chat/stream")
async def chat_with_documents_stream(chat_request: ChatRequest, request: Request):
    """Streaming chat interface with document context using Qwen - No authentication required"""

    async def generate_response():
        try:
            from ollama_client import ollama_client

            logger.info(f"Processing streaming chat request")

            # Search for relevant documents
            context_chunks, context_info = await search_relevant_documents(
                chat_request.message,
                chat_request.categories,
                threshold=0.7  # Slightly higher threshold for streaming
            )
            
            # Extract context metadata for tracking
            context_metadata = context_info

            # Send initial metadata (will be updated later with filtered references)
            metadata_response = {
                "type": "metadata",
                "conversation_id": chat_request.conversation_id or str(uuid.uuid4()),
                "context_chunks": context_info,
                "context_found": len(context_chunks) > 0,
                "model_used": ollama_client.chat_model,
            }
            yield f"data: {json.dumps(metadata_response)}\n\n"

            # Prepare system prompt for Qwen3
            base_system_prompt = (
                f"You are Jagruti, a helpful assistant for IIIT Hyderabad. "
                f"You help students, faculty, and staff find information from official documents. "
                f"Use the provided context (if it is useful) to answer questions accurately.  Don't mention the word context though. "
                f"If you cannot find relevant information in the context, say so politely. "
                f"When answering, naturally mention the document names you are referencing (e.g., 'According to [document name]...'). "
                f"Always be helpful, concise, and reference the source documents when applicable. "
                f"Today's datetime is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
            )

            # Build conversation context from client-provided history
            conversation_context = ""
            if chat_request.conversation_history and len(chat_request.conversation_history) > 0:
                conversation_context = "\n\nPrevious conversation:\n"
                # Use last 10 messages for context
                for msg in chat_request.conversation_history[-10:]:
                    role = "Human" if msg.type == "user" else "Assistant"
                    conversation_context += f"{role}: {msg.content}\n"
                conversation_context += "Please maintain context from this conversation."

            # Build full system prompt with conversation history
            full_system_prompt = base_system_prompt + conversation_context

            # Collect the full response to determine referenced files
            full_response = ""
            
            # Generate streaming response
            async for chunk in ollama_client.generate_response_stream(
                prompt=chat_request.message,
                context=context_chunks,
                system_prompt=full_system_prompt,
                context_metadata=context_metadata
            ):
                if chunk:
                    full_response += chunk
                    chunk_response = {
                        "type": "content",
                        "content": chunk,
                        "is_final": False
                    }
                    yield f"data: {json.dumps(chunk_response)}\n\n"

            # Extract which files were actually referenced in the response
            available_files = [meta.get('filename', 'Unknown') for meta in context_metadata]
            referenced_files = ollama_client.extract_referenced_files(full_response, available_files)
            
            # Filter context_info to only include referenced files
            filtered_context_info = [
                info for info in context_info 
                if info.get('filename') in referenced_files
            ]
            
            logger.info(f"Stream - Original context: {len(context_info)} files, Referenced: {len(filtered_context_info)} files")
            
            # Send updated metadata with filtered references
            updated_metadata_response = {
                "type": "metadata_update",
                "context_chunks": filtered_context_info,
            }
            yield f"data: {json.dumps(updated_metadata_response)}\n\n"

            # Send final marker
            final_response = {
                "type": "content",
                "content": "",
                "is_final": True
            }
            yield f"data: {json.dumps(final_response)}\n\n"

        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            error_response = {
                "type": "error",
                "error": ("I apologize, but I'm having trouble processing "
                          "your request right now. Please try again later.")
            }
            yield f"data: {json.dumps(error_response)}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Transfer-Encoding": "chunked",
        }
    )


@app.post("/api/debug/test-conversation")
async def debug_test_conversation(chat_request: ChatRequest):
    """Debug endpoint to test conversation history handling"""
    try:
        logger.info(f"Test conversation request: {chat_request.message}")
        logger.info(
            f"Conversation history: {len(chat_request.conversation_history or [])} messages")

        # Build system prompt with conversation history (same logic as main chat)
        base_system_prompt = (
            "You are Jagruti, a helpful assistant. "
            "You can remember information from previous messages in this conversation."
        )

        system_prompt = base_system_prompt
        if chat_request.conversation_history and len(chat_request.conversation_history) > 0:
            history_text = "\n\nPrevious conversation:\n"
            for msg in chat_request.conversation_history[-10:]:
                role = "Human" if msg.type == "user" else "Assistant"
                history_text += f"{role}: {msg.content}\n"
            system_prompt = base_system_prompt + history_text + \
                "\nRemember the context from this conversation."

        return {
            "message": chat_request.message,
            "conversation_history_count": len(chat_request.conversation_history or []),
            "system_prompt_preview": system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt,
            "full_system_prompt_length": len(system_prompt),
        }

    except Exception as e:
        logger.error(f"Error in test conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/debug/search-relevance")
async def debug_search_relevance(
    query: str,
    threshold: Optional[float] = None,
    limit: int = 10,
    current_user: UserInfo = Depends(get_current_user)
):
    """Debug endpoint to check search relevance filtering"""
    try:
        # Use provided threshold or default
        relevance_threshold = threshold if threshold is not None else float(
            os.getenv("RELEVANCE_THRESHOLD", "0.7"))

        # Perform search without category filtering for debugging
        search_results = chunks_collection.query(
            query_texts=[query],
            n_results=limit,
        )

        results = {
            "query": query,
            "threshold": relevance_threshold,
            "all_results": [],
            "filtered_results": [],
            "stats": {
                "total_found": 0,
                "after_filtering": 0,
                "excluded_count": 0,
            }
        }

        if search_results["documents"] and search_results["distances"]:
            total_found = len(search_results["documents"][0])
            results["stats"]["total_found"] = total_found

            for i, (doc, metadata, distance) in enumerate(zip(
                search_results["documents"][0],
                search_results["metadatas"][0],
                search_results["distances"][0]
            )):
                result_item = {
                    "rank": i + 1,
                    "distance": round(distance, 4),
                    "relevance_score": round(1.0 - distance, 4),
                    "filename": metadata.get("filename", "Unknown"),
                    "category": metadata.get("category", "Unknown"),
                    "doc_id": metadata.get("doc_id", ""),
                    "chunk_preview": doc[:100] + "..." if len(doc) > 100 else doc,
                    "included": distance <= relevance_threshold,
                }

                results["all_results"].append(result_item)

                if distance <= relevance_threshold:
                    results["filtered_results"].append(result_item)
                else:
                    results["stats"]["excluded_count"] += 1

            results["stats"]["after_filtering"] = len(
                results["filtered_results"])

        return results

    except Exception as e:
        logger.error(f"Error in debug search relevance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/categories")
async def get_categories(current_user: UserInfo = Depends(get_current_user)):
    """Get available document categories"""
    return {
        "categories": [
            {"value": "faculty", "label": "Faculty Data"},
            {"value": "student", "label": "Student Data"},
            {"value": "hostel", "label": "Hostels"},
            {"value": "academics", "label": "Academics"},
            {"value": "mess", "label": "Messes"},
        ]
    }


@app.get("/api/stats")
async def get_stats(current_user: UserInfo = Depends(get_current_user)):
    """Get system statistics"""
    try:
        doc_count = documents_collection.count()
        chunk_count = chunks_collection.count()

        return {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "categories": {
                "faculty": 0,  # TODO: Count by category
                "student": 0,
                "hostel": 0,
                "academics": 0,
                "mess": 0,
            },
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"total_documents": 0, "total_chunks": 0, "categories": {}}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8001)),
        reload=os.getenv("DEBUG", "True").lower() == "true",
    )
