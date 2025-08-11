from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import uuid
import httpx
from urllib.parse import urlencode, quote
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="askIIIT API",
    description="Backend API for IIIT document search and chatbot",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
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
    category: Optional[str] = None
    limit: Optional[int] = 10


class ChatRequest(BaseModel):
    message: str
    category: Optional[str] = None
    conversation_id: Optional[str] = None


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
                username = response.text.split("<cas:user>")[1].split("</cas:user>")[0]

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
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return current_user


# Initialize ChromaDB collections
try:
    documents_collection = chroma_client.get_or_create_collection(
        name="documents", metadata={"description": "Document metadata collection"}
    )
    chunks_collection = chroma_client.get_or_create_collection(
        name="chunks", metadata={"description": "Document chunks with embeddings"}
    )
    logger.info("ChromaDB collections initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize collections: {e}")
    raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "askIIIT API is running!",
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

        # Check Ollama connection
        try:
            ollama_available = await ollama_client.check_connection()
            if ollama_available:
                models = await ollama_client.list_models()
                health_status["services"]["ollama"] = {
                    "status": "healthy",
                    "available_models": [model["name"] for model in models],
                    "chat_model": ollama_client.chat_model,
                }
            else:
                health_status["services"]["ollama"] = {
                    "status": "unhealthy",
                    "error": "Cannot connect to Ollama server",
                }
        except Exception as e:
            health_status["services"]["ollama"] = {
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
                    "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"
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

        logger.info(f"Document {file.filename} processed successfully: {result}")

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
    category: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: UserInfo = Depends(get_current_user),
):
    """List documents with optional filtering"""
    try:
        # Query documents collection
        where_clause = {}
        if category:
            where_clause["category"] = category

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
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@app.post("/api/search")
async def search_documents(
    query_request: QueryRequest, current_user: UserInfo = Depends(get_current_user)
):
    """Search documents using semantic search"""
    try:
        # Build where clause for filtering
        where_clause = {}
        if query_request.category:
            where_clause["category"] = query_request.category

        # Perform semantic search on chunks
        results = chunks_collection.query(
            query_texts=[query_request.query],
            n_results=query_request.limit,
            where=where_clause if where_clause else None,
        )

        return {
            "query": query_request.query,
            "results": [
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
                for chunk_id, document, metadata, distance in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ],
        }
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/api/chat")
async def chat_with_documents(
    chat_request: ChatRequest, current_user: UserInfo = Depends(get_current_user)
):
    """Chat interface with document context using Qwen3"""
    try:
        # Import Ollama client
        from ollama_client import ollama_client

        # First, search for relevant documents
        where_clause = {}
        if chat_request.category:
            where_clause["category"] = chat_request.category

        search_results = chunks_collection.query(
            query_texts=[chat_request.message],
            n_results=5,
            where=where_clause if where_clause else None,
        )

        context_chunks = (
            search_results["documents"][0] if search_results["documents"] else []
        )
        context_metadata = (
            search_results["metadatas"][0] if search_results["metadatas"] else []
        )

        # Prepare system prompt for Qwen3
        system_prompt = (
            "You are askIIIT, a helpful assistant for IIIT Hyderabad. "
            "You help students, faculty, and staff find information from official documents. "
            "Use the provided context to answer questions accurately. "
            "If you cannot find relevant information in the context, say so politely. "
            "Always be helpful, concise, and reference the source documents when applicable."
        )

        # Generate response using Qwen3 via Ollama
        if context_chunks:
            response = await ollama_client.generate_response(
                prompt=chat_request.message,
                context=context_chunks,
                system_prompt=system_prompt,
            )
        else:
            response = await ollama_client.generate_response(
                prompt=chat_request.message,
                system_prompt=system_prompt
                + " Note: No relevant documents were found for this query.",
            )

        # Prepare context information for frontend
        context_info = []
        for chunk, metadata in zip(context_chunks, context_metadata):
            context_info.append(
                {
                    "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    "filename": metadata.get("filename", "Unknown"),
                    "category": metadata.get("category", "Unknown"),
                    "doc_id": metadata.get("doc_id", ""),
                    "relevance_score": 1.0,  # You could calculate this from distances
                }
            )

        return {
            "message": chat_request.message,
            "response": response,
            "context_chunks": context_info,
            "context_found": len(context_chunks) > 0,
            "conversation_id": chat_request.conversation_id or str(uuid.uuid4()),
            "model_used": "qwen3:8b",
        }

    except Exception as e:
        logger.error(f"Error in chat: {e}")

        # Fallback response
        fallback_response = (
            "I apologize, but I'm having trouble processing your request right now. "
            "This might be because the Qwen3 model is not available or there's a connection issue. "
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
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "True").lower() == "true",
    )
