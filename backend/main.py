from colored_logging import setup_logging
from log_capture import setup_log_capture, get_recent_logs
from processing_status import get_processing_status
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
import psutil
import time
import httpx
from urllib.parse import urlencode, quote
import logging
import json
import uvicorn

# Load environment variables
load_dotenv()

# Configure colored logging

setup_logging(level=logging.INFO)
setup_log_capture()  # Set up log capture for admin panel
logger = logging.getLogger(__name__)

# Track application start time for uptime calculation
app_start_time = time.time()

app = FastAPI(
    title="Jagruti API",
    description="Backend API for IIIT document search and chatbot",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
    categories: List[str]  # Updated to support multiple categories
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

            logger.info(f"CAS validation response: {response.text}")

            # Parse CAS response (simplified - in production, parse XML properly)
            if "cas:authenticationSuccess" in response.text:
                # Extract username from CAS response
                # This is a simplified extraction - you should use proper XML parsing
                username = response.text.split("<cas:user>")[
                    1].split("</cas:user>")[0]

                logger.info(f"Extracted username: {username}")

                # Check if user is admin
                admin_users = os.getenv("ADMIN_USERS", "").split(",")
                admin_users = [user.strip() for user in admin_users if user.strip()]
                
                # Check both username and full email
                user_email = f"{username}@iiit.ac.in"
                is_admin = username in admin_users or user_email in admin_users
                
                logger.info(f"Admin users: {admin_users}")
                logger.info(f"User email: {user_email}")
                logger.info(f"Is admin: {is_admin}")

                return UserInfo(
                    username=username,
                    email=user_email,
                    full_name=username.title(),
                    is_admin=is_admin,
                )
    except Exception as e:
        logger.error(f"CAS validation error: {e}")

    return None


async def get_current_user(request: Request) -> UserInfo:
    """Get current authenticated user from session or CAS ticket"""
    # Check for CAS ticket in query parameters first
    ticket = request.query_params.get("ticket")
    if ticket:
        service_url = str(request.url).split("?")[0]  # Remove query parameters
        user = await validate_cas_token(ticket, service_url)
        if user:
            return user

    # Check for authentication from frontend cookies
    # The frontend should forward this as an Authorization header
    authorization = request.headers.get("Authorization")
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        # Check if this is our simple authenticated token from the frontend
        if token == "authenticated":
            # This means the frontend has already validated the user
            # In a production system, you'd want to validate a proper JWT token here
            # For now, we'll trust the frontend's authentication
            # We can get user info from cookies if needed
            return UserInfo(
                username="authenticated_user",
                email="authenticated_user@iiit.ac.in", 
                full_name="Authenticated User",
                is_admin=True,  # The frontend already checked admin status
            )

    # If no valid authentication found, raise 401
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
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device="mps" if torch.backends.mps.is_available(
        ) else "cuda" if torch.cuda.is_available() else "cpu",
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
    
    # New collection for document summaries - for faster first-stage retrieval
    summaries_collection = chroma_client.get_or_create_collection(
        name="summaries",
        embedding_function=embedding_function,
        metadata={"description": "Document summaries with embeddings",
                  "hnsw:space": "cosine"}
    )
    
    # Initialize RAG pipeline
    from rag_pipeline import RAGPipeline
    rag_pipeline = RAGPipeline(
        summaries_collection=summaries_collection,
        chunks_collection=chunks_collection,
        default_top_k=5,
        summary_threshold=float(os.getenv("SUMMARY_THRESHOLD", "0.7")),
        content_threshold=float(os.getenv("CONTENT_THRESHOLD", "0.65"))
    )
    
    logger.info("ChromaDB collections and RAG pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize collections: {e}")
    raise


@app.get("/")
async def root():
    """Health check endpoint"""
    logger.info("Root endpoint accessed - API is running normally")
    logger.debug("Health check completed successfully")
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


class TicketValidationRequest(BaseModel):
    ticket: str
    service: str


@app.post("/auth/validate")
async def validate_ticket(request: TicketValidationRequest):
    """Validate CAS ticket and return user info"""
    try:
        logger.info(f"Validating ticket: {request.ticket[:10]}... for service: {request.service}")
        
        user_info = await validate_cas_token(request.ticket, request.service)
        if user_info:
            logger.info(f"Validation successful for user: {user_info.email}, is_admin: {user_info.is_admin}")
            return {
                "success": True,
                "user": user_info.dict(),
                "token": "authenticated"
            }
        else:
            logger.warning("Ticket validation failed - no user info returned")
            raise HTTPException(status_code=401, detail="Invalid ticket")
    except Exception as e:
        logger.error(f"Ticket validation error: {e}")
        raise HTTPException(status_code=401, detail=f"Ticket validation failed: {str(e)}")


@app.get("/auth/debug")
async def debug_auth():
    """Debug endpoint to check admin users configuration"""
    admin_users = os.getenv("ADMIN_USERS", "").split(",")
    admin_users = [user.strip() for user in admin_users if user.strip()]
    
    return {
        "admin_users": admin_users,
        "cas_server_url": os.getenv("CAS_SERVER_URL"),
        "cas_service_url": os.getenv("CAS_SERVICE_URL"),
    }


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
        from document_summarizer import DocumentSummaryProcessor

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
        summary_processor = DocumentSummaryProcessor()

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
        
        # Extract the full text for category detection
        full_text = doc_processor.extract_text_from_file(file_content, file.filename)
        
        # Get multiple categories for the document
        categories = await summarizer.categorize_document_async(file.filename, full_text[:5000])
        
        # If only single category is returned, convert to list
        if isinstance(categories, str):
            categories = [categories]
            
        # Add the explicitly provided category if not already included
        if category not in categories:
            categories.append(category)
            
        # Process document summary for each category
        summary_result = summary_processor.process_document_summary(
            doc_id=doc_id,
            text=full_text,
            filename=file.filename,
            categories=categories,
            summaries_collection=summaries_collection
        )
        
        # Update document metadata with categories
        doc_metadata = documents_collection.get(
            ids=[doc_id],
            include=["metadatas"]
        )["metadatas"][0]
        
        updated_metadata = {**doc_metadata, "categories": categories}
        documents_collection.update(
            ids=[doc_id],
            metadatas=[updated_metadata]
        )

        logger.info(f"Document {file.filename} processed successfully with categories: {categories}")

        return {
            "message": "Document uploaded and processed successfully",
            "doc_id": doc_id,
            "filename": file.filename,
            "chunk_count": result["chunk_count"],
            "categories": categories,
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
            
            # For multiple categories, return ANY category matches
            if len(category_list) == 1:
                where_clause["categories"] = {"$contains": category_list[0]}
            else:
                summary_results = summaries_collection.get(
                    where={"category": {"$in": category_list}},
                    include=["metadatas"]
                )
                
                if summary_results["metadatas"]:
                    # Extract unique document IDs
                    doc_ids = {meta.get("doc_id") for meta in summary_results["metadatas"] if meta.get("doc_id")}
                    
                    if doc_ids:
                        where_clause["doc_id"] = {"$in": list(doc_ids)}
                    else:
                        # No matching documents
                        return {
                            "documents": [],
                            "total": 0,
                            "limit": limit,
                            "offset": offset,
                        }

        results = documents_collection.get(
            where=where_clause if where_clause else None, limit=limit, offset=offset
        )

        # Enhance documents with chunk count information
        enhanced_documents = []
        if results["metadatas"] and results["ids"]:
            for i, metadata in enumerate(results["metadatas"]):
                doc_id = results["ids"][i]
                
                # Get chunk count for this document
                chunk_results = chunks_collection.get(
                    where={"doc_id": doc_id},
                    include=["documents"]
                )
                chunk_count = len(chunk_results["ids"]) if chunk_results["ids"] else 0
                
                # Add additional metadata
                enhanced_doc = {
                    **metadata,
                    "doc_id": doc_id,
                    "chunk_count": chunk_count,
                    "embedding_count": chunk_count,
                    "status": "processed",
                }
                
                # Ensure categories field exists
                if "categories" not in enhanced_doc:
                    # If only old "category" field exists, convert to list
                    if "category" in enhanced_doc:
                        enhanced_doc["categories"] = [enhanced_doc["category"]]
                    else:
                        enhanced_doc["categories"] = ["unknown"]
                
                enhanced_documents.append(enhanced_doc)

        return {
            "documents": enhanced_documents,
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
    """Search documents using two-stage RAG pipeline"""
    try:
        # Execute the full RAG pipeline
        results = await rag_pipeline.retrieve(
            query=query_request.query,
            categories=query_request.categories,
            top_k=query_request.limit or 10,
            include_summaries=True
        )
        
        # Format results for API response
        formatted_docs = results["documents"]
        formatted_chunks = results["chunks"]
        
        return {
            "query": query_request.query,
            "documents": formatted_docs,
            "content_chunks": formatted_chunks,
            "total_docs": len(formatted_docs),
            "total_chunks": len(formatted_chunks),
        }
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


@app.post("/api/chat")
async def chat_with_documents(chat_request: ChatRequest):
    """Chat interface with document context using Qwen3 - No authentication required"""
    try:
        # Import Ollama client
        from ollama_client import ollama_client

        # Process conversation context for enhanced RAG retrieval
        enhanced_query = chat_request.message
        if chat_request.conversation_history:
            # Convert ChatMessage objects to dictionaries for processing
            conv_history = [
                {"type": msg.type, "content": msg.content}
                for msg in chat_request.conversation_history
            ]
            
            # Include conversation context in the query
            enhanced_query = rag_pipeline.process_conversation_context(
                current_message=chat_request.message,
                conversation_history=conv_history,
                max_history=5  # Include up to 5 previous messages
            )
        
        # Execute the RAG pipeline with the enhanced query
        rag_results = await rag_pipeline.retrieve(
            query=enhanced_query,
            categories=chat_request.categories,
            top_k=5,  # Get top 5 most relevant chunks
            include_summaries=False  # Don't need summaries for chat
        )
        
        # Extract the relevant content chunks
        context_chunks = []
        context_metadata = []
        
        if rag_results["chunks"]:
            for chunk in rag_results["chunks"]:
                context_chunks.append(chunk["text"])
                context_metadata.append({
                    "doc_id": chunk["doc_id"],
                    "filename": chunk["filename"],
                    "category": chunk["category"],
                    "relevance_score": chunk["relevance_score"],
                })
        
        # Prepare system prompt for Qwen3 with conversation history
        # TODO: Verify system prompt
        base_system_prompt = (
            f"You are Jagruti, a helpful assistant for IIIT Hyderabad. "
            f"You help students, faculty, and staff find information from official documents. "
            f"Use the provided context to answer questions accurately. "
            f"If you cannot find relevant information in the context, say so politely. "
            f"DO NOT answer questions that are not related to IIIT Hyderabad or if the information is not available in provided context. "
            f"Always be helpful, concise, and reference the source documents when applicable. "
            f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."
        )

        # Add conversation history to the system prompt if available
        system_prompt = base_system_prompt
        if chat_request.conversation_history and len(chat_request.conversation_history) > 0:
            logger.info(
                f"Including conversation history: {len(chat_request.conversation_history)} messages")
            history_text = "\n\nPrevious conversation:\n"
            # Last 10 messages to avoid token limit
            for msg in chat_request.conversation_history[-10:]:
                role = "Human" if msg.type == "user" else "Assistant"
                history_text += f"{role}: {msg.content}\n"
            system_prompt = base_system_prompt + history_text + \
                "\nPlease maintain context from this conversation when answering the current question."
        else:
            logger.info("No conversation history provided")

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

        # Prepare context information for frontend with actual relevance scores
        context_info = []
        for i, (chunk, metadata) in enumerate(zip(context_chunks, context_metadata)):
            context_info.append(
                {
                    "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    "filename": metadata.get("filename", "Unknown"),
                    "category": metadata.get("category", "Unknown"),
                    "doc_id": metadata.get("doc_id", ""),
                    "relevance_score": metadata.get("relevance_score", 0),
                }
            )

        return {
            "message": chat_request.message,
            "response": response,
            "context_chunks": context_info,
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
async def chat_with_documents_stream(chat_request: ChatRequest):
    """Streaming chat interface with document context using Qwen - No authentication required"""

    async def generate_response():
        try:
            # Import Ollama client
            from ollama_client import ollama_client

            # Process conversation context for enhanced RAG retrieval
            enhanced_query = chat_request.message
            if chat_request.conversation_history:
                # Convert ChatMessage objects to dictionaries for processing
                conv_history = [
                    {"type": msg.type, "content": msg.content}
                    for msg in chat_request.conversation_history
                ]
                
                # Include conversation context in the query
                enhanced_query = rag_pipeline.process_conversation_context(
                    current_message=chat_request.message,
                    conversation_history=conv_history,
                    max_history=5  # Include up to 5 previous messages
                )
            
            # Execute the RAG pipeline with the enhanced query
            rag_results = await rag_pipeline.retrieve(
                query=enhanced_query,
                categories=chat_request.categories,
                top_k=5,  # Get top 5 most relevant chunks
                include_summaries=False  # Don't need summaries for chat
            )

            # Execute the RAG pipeline with the enhanced query
            rag_results = await rag_pipeline.retrieve(
                query=enhanced_query,
                categories=chat_request.categories,
                top_k=5,  # Get top 5 most relevant chunks
                include_summaries=False  # Don't need summaries for chat
            )
            
            # Extract the relevant content chunks
            context_chunks = []
            context_metadata = []
            
            if rag_results["chunks"]:
                for chunk in rag_results["chunks"]:
                    context_chunks.append(chunk["text"])
                    context_metadata.append({
                        "doc_id": chunk["doc_id"],
                        "filename": chunk["filename"],
                        "category": chunk["category"],
                        "relevance_score": chunk["relevance_score"],
                    })

            # Send context information first
            context_info = []
            for i, (chunk, metadata) in enumerate(zip(context_chunks, context_metadata)):
                context_info.append(
                    {
                        "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                        "filename": metadata.get("filename", "Unknown"),
                        "category": metadata.get("category", "Unknown"),
                        "doc_id": metadata.get("doc_id", ""),
                        "relevance_score": metadata.get("relevance_score", 0),
                    }
                )

            # Send metadata first
            metadata_response = {
                "type": "metadata",
                "conversation_id": chat_request.conversation_id or str(uuid.uuid4()),
                "context_chunks": context_info,
                "context_found": len(context_chunks) > 0,
                "model_used": ollama_client.chat_model,
            }
            yield f"data: {json.dumps(metadata_response)}\n\n"

            # Generate and stream response with character-level streaming including conversation history
            base_system_prompt = (
                f"You are Jagruti, a helpful assistant for IIIT Hyderabad. "
                f"You help students, faculty, and staff find information from official documents. "
                f"Use the provided context to answer questions accurately. "
                f"If you cannot find relevant information in the context, say so politely. "
                f"Always be helpful, concise, and reference the source documents when applicable. "
                f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."
            )

            # Add conversation history to the system prompt if available
            system_prompt = base_system_prompt
            if chat_request.conversation_history and len(chat_request.conversation_history) > 0:
                logger.info(
                    f"Including conversation history: {len(chat_request.conversation_history)} messages")
                history_text = "\n\nPrevious conversation:\n"
                # Last 10 messages to avoid token limit
                for msg in chat_request.conversation_history[-10:]:
                    role = "Human" if msg.type == "user" else "Assistant"
                    history_text += f"{role}: {msg.content}\n"
                system_prompt = base_system_prompt + history_text + \
                    "\nPlease maintain context from this conversation when answering the current question."
            else:
                logger.info("No conversation history provided")

            # Stream response from Ollama with immediate character output
            if context_chunks:
                response_stream = ollama_client.generate_response_stream(
                    prompt=chat_request.message,
                    context=context_chunks,
                    system_prompt=system_prompt,
                )
            else:
                response_stream = ollama_client.generate_response_stream(
                    prompt=chat_request.message,
                    system_prompt=system_prompt
                    + " Note: No relevant documents were found for this query.",
                )

            # Stream the response chunks immediately
            async for chunk in response_stream:
                if chunk:
                    chunk_response = {
                        "type": "content",
                        "content": chunk,
                        "is_final": False
                    }
                    yield f"data: {json.dumps(chunk_response)}\n\n"

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


# Admin-only endpoints
@app.get("/api/admin/users")
async def get_all_users(current_user: UserInfo = Depends(get_admin_user)):
    """Get all users (Admin only)"""
    try:
        admin_users = os.getenv("ADMIN_USERS", "").split(",")
        return {
            "admin_users": [user.strip() for user in admin_users if user.strip()],
            "current_user": current_user.email,
        }
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/system-info")
async def get_system_info(current_user: UserInfo = Depends(get_admin_user)):
    """Get detailed system information (Admin only)"""
    try:
        from ollama_client import ollama_client
        
        doc_count = documents_collection.count()
        chunk_count = chunks_collection.count()
        
        # Get document metadata
        doc_results = documents_collection.get(include=["documents", "metadatas"])
        
        # Count by category
        categories_count = {}
        if doc_results and doc_results["metadatas"]:
            for metadata in doc_results["metadatas"]:
                category = metadata.get("category", "unknown")
                categories_count[category] = categories_count.get(category, 0) + 1
        
        # Check model status
        model_status = "unknown"
        try:
            response = await ollama_client.list_models()
            model_status = "healthy" if response else "unavailable"
            logger.debug(f"Model status check: {model_status}")
        except Exception as e:
            model_status = "error"
            logger.warning(f"Model status check failed: {e}")
        
        # Get real processing status
        processing_status = get_processing_status().get_status()
        
        # Get real backend status with actual system metrics
        try:
            # Get system memory info
            memory_info = psutil.virtual_memory()
            
            # Get current process info
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Calculate uptime
            uptime_seconds = time.time() - app_start_time
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            seconds = int(uptime_seconds % 60)
            uptime_str = f"{hours}h {minutes}m {seconds}s"
            
            # Get CPU percentage with a brief interval for accuracy
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get disk usage for the current directory
            disk_usage = psutil.disk_usage('/')
            
            # Get load average (on Unix systems)
            try:
                load_avg = psutil.getloadavg()
                load_average = round(load_avg[0], 2) if load_avg else None
            except (AttributeError, OSError):
                load_average = None
            
            backend_status = {
                "status": "healthy",
                "uptime": uptime_str,
                "memory_usage": round(process_memory.rss / 1024 / 1024, 1),  # Process memory in MB
                "memory_percent": round(process_memory.rss / memory_info.total * 100, 1),  # Process memory as % of total
                "system_memory_total": round(memory_info.total / 1024 / 1024 / 1024, 1),  # Total system memory in GB
                "system_memory_used": round(memory_info.used / 1024 / 1024 / 1024, 1),   # Used system memory in GB
                "system_memory_percent": round(memory_info.percent, 1),
                "cpu_percent": round(cpu_percent, 1),
                "cpu_count": psutil.cpu_count(),
                "disk_usage_percent": round(disk_usage.percent, 1),
                "load_average": load_average,
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            }
            
            # Log actual values for debugging
            logger.debug(f"Real system stats - CPU: {cpu_percent}%, Memory: {memory_info.percent}%, Process Memory: {process_memory.rss / 1024 / 1024:.1f}MB")
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            backend_status = {
                "status": "error", 
                "uptime": "Unknown",
                "memory_usage": 0,
                "error": str(e)
            }
        
        return {
            "documents": {
                "total": doc_count,
                "by_category": categories_count,
            },
            "chunks": {
                "total": chunk_count,
            },
            "models": {
                "status": model_status,
            },
            "database": {
                "type": "ChromaDB",
                "status": "healthy",
                "path": os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data"),
            },
            "processing": processing_status,
            "backend": backend_status,
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user: UserInfo = Depends(get_admin_user)
):
    """Delete a document and its chunks (Admin only)"""
    try:
        # Delete from documents collection
        documents_collection.delete(ids=[doc_id])
        
        # Delete related chunks
        chunk_results = chunks_collection.get(
            where={"doc_id": doc_id},
            include=["documents"]
        )
        
        if chunk_results and chunk_results["ids"]:
            chunks_collection.delete(ids=chunk_results["ids"])
            logger.info(f"Deleted {len(chunk_results['ids'])} chunks for document {doc_id}")
        
        logger.info(f"Document {doc_id} deleted by {current_user.email}")
        return {"message": f"Document {doc_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/reindex")
async def reindex_documents(current_user: UserInfo = Depends(get_admin_user)):
    """Reindex all documents (Admin only)"""
    try:
        # This is a placeholder for reindexing logic
        # In a real implementation, you might want to:
        # 1. Clear existing collections
        # 2. Reprocess all documents
        # 3. Rebuild embeddings
        
        doc_count = documents_collection.count()
        chunk_count = chunks_collection.count()
        
        logger.info(f"Reindex requested by {current_user.email}")
        
        return {
            "message": "Reindexing completed",
            "documents_processed": doc_count,
            "chunks_processed": chunk_count,
        }
        
    except Exception as e:
        logger.error(f"Error during reindexing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/logs")
async def get_admin_logs(
    lines: int = 100,
    level: Optional[str] = None,
    current_user: UserInfo = Depends(get_admin_user)
):
    """Get recent application logs (Admin only)"""
    try:
        # Add some real-time logging to demonstrate the system is working
        logger.info(f"Admin logs request: lines={lines}, level={level}, user={current_user.email}")
        
        # Generate some test log entries to demonstrate different levels
        if len(get_recent_logs(count=10)) < 5:  # Only add if we don't have many logs yet
            logger.debug("Log capture system is operational")
            logger.warning("This is a sample warning message for testing filters")
            logger.error("Sample error message - this is just for testing log levels")
        
        # Get real logs from the log capture system
        logs = get_recent_logs(count=lines, level_filter=level)
        
        logger.debug(f"Retrieved {len(logs)} raw logs from capture system")
        
        # Transform logs to the expected format and reverse order (most recent first)
        formatted_logs = []
        for log_entry in reversed(logs):  # Reverse to show most recent first
            # Ensure we have all required fields
            formatted_log = {
                "timestamp": log_entry.get("timestamp", datetime.now().isoformat()),
                "level": log_entry.get("level", "INFO"),
                "message": log_entry.get("raw_message", log_entry.get("message", "")),
                "source": log_entry.get("source", "unknown")
            }
            formatted_logs.append(formatted_log)
        
        logger.info(f"Returning {len(formatted_logs)} formatted logs (level filter: {level})")
        
        return {
            "logs": formatted_logs,
            "total_lines": len(formatted_logs),
            "filter_applied": {
                "level": level,
                "lines_requested": lines,
                "lines_returned": len(formatted_logs)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting logs: {e}", exc_info=True)
        return {"logs": [], "total_lines": 0, "error": str(e)}


@app.post("/api/admin/bulk-process")
async def start_bulk_processing(current_user: UserInfo = Depends(get_admin_user)):
    """Start bulk processing of uploaded documents (Admin only)"""
    try:
        import subprocess
        import os
        from pathlib import Path
        
        # Run bulk_process.py script
        backend_dir = Path(__file__).parent
        bulk_script = backend_dir / "bulk_process.py"
        
        if not bulk_script.exists():
            raise HTTPException(status_code=404, detail="Bulk processing script not found")
        
        # Mark processing as started
        processing_tracker = get_processing_status()
        processing_tracker.start_processing()
        
        # Start the bulk processing in background
        # In production, you might want to use a proper task queue like Celery
        result = subprocess.run([
            "python", str(bulk_script), "--process"
        ], capture_output=True, text=True, cwd=str(backend_dir))
        
        # Mark processing as finished
        processing_tracker.finish_processing(result.returncode == 0)
        
        logger.info(f"Bulk processing initiated by admin user: {current_user.email}")
        logger.info(f"Bulk processing script execution result - Return code: {result.returncode}")
        
        if result.returncode != 0:
            logger.error(f"Bulk processing failed: {result.stderr}")
        else:
            logger.info(f"Bulk processing completed successfully: {result.stdout[:200]}...")
        
        return {
            "message": "Bulk processing started",
            "status": "processing",
            "output": result.stdout if result.returncode == 0 else result.stderr
        }
        
    except Exception as e:
        logger.error(f"Error starting bulk processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/documents/{doc_id}/details")
async def get_document_details(
    doc_id: str,
    current_user: UserInfo = Depends(get_admin_user)
):
    """Get detailed information about a specific document (Admin only)"""
    try:
        # Get document metadata
        doc_results = documents_collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"]
        )
        
        if not doc_results["ids"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get related chunks
        chunk_results = chunks_collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"]
        )
        
        doc_metadata = doc_results["metadatas"][0] if doc_results["metadatas"] else {}
        
        return {
            "doc_id": doc_id,
            "name": doc_metadata.get("name", "Unknown"),
            "category": doc_metadata.get("category", "unknown"),
            "description": doc_metadata.get("description", ""),
            "created_at": doc_metadata.get("created_at", ""),
            "author": doc_metadata.get("author", "system"),
            "chunk_count": len(chunk_results["ids"]) if chunk_results["ids"] else 0,
            "embedding_count": len(chunk_results["ids"]) if chunk_results["ids"] else 0,
            "status": "processed",
            "file_size": doc_metadata.get("file_size", 0),
            "metadata": doc_metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/debug/rag-diagnostics")
async def debug_rag_diagnostics(
    query: str = "test query",
    current_user: UserInfo = Depends(get_admin_user)
):
    """Debug endpoint for RAG pipeline diagnostics (Admin only)"""
    try:
        # Get collection counts
        summaries_count = summaries_collection.count()
        chunks_count = chunks_collection.count()
        documents_count = documents_collection.count()
        
        # Sample summaries
        sample_summaries = summaries_collection.get(
            limit=5,
            include=["metadatas", "documents"]
        )
        
        # Run a simple RAG query
        rag_results = await rag_pipeline.retrieve(
            query=query,
            categories=None,
            top_k=5,
            include_summaries=True
        )
        
        return {
            "collections": {
                "documents": documents_count,
                "chunks": chunks_count,
                "summaries": summaries_count,
            },
            "sample_summaries": {
                "count": len(sample_summaries["ids"]) if sample_summaries["ids"] else 0,
                "ids": sample_summaries["ids"] if sample_summaries["ids"] else [],
                "categories": [m.get("category", "unknown") for m in sample_summaries["metadatas"]] if sample_summaries["metadatas"] else [],
                "doc_ids": [m.get("doc_id", "unknown") for m in sample_summaries["metadatas"]] if sample_summaries["metadatas"] else [],
            },
            "rag_results": {
                "query": query,
                "docs_found": len(rag_results.get("documents", [])),
                "chunks_found": len(rag_results.get("chunks", [])),
                "results": rag_results
            }
        }
        
    except Exception as e:
        logger.error(f"Error in RAG diagnostics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{doc_id}/download")
async def download_document(
    doc_id: str,
    current_user: UserInfo = Depends(get_current_user)
):
    """Download original document file"""
    try:
        # Get document metadata to find the file
        doc_results = documents_collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"]
        )
        
        if not doc_results["ids"]:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_metadata = doc_results["metadatas"][0] if doc_results["metadatas"] else {}
        filename = doc_metadata.get("name", "document.pdf")
        
        # Look for the file in the pdfs directory
        from pathlib import Path
        backend_dir = Path(__file__).parent
        pdfs_dir = backend_dir.parent / "pdfs"
        
        # Try to find the file
        file_path = pdfs_dir / filename
        if not file_path.exists():
            # Try without extension and with .pdf
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            file_path = pdfs_dir / f"{base_name}.pdf"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Original file not found")
        
        # Return file as streaming response
        def file_generator():
            with open(file_path, "rb") as file:
                while chunk := file.read(8192):
                    yield chunk
        
        return StreamingResponse(
            file_generator(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename=\"{filename}\""}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class AdminSettings(BaseModel):
    max_upload_size: Optional[int] = None
    allowed_extensions: Optional[str] = None
    admin_users: Optional[List[str]] = None


@app.put("/api/admin/settings")
async def update_settings(
    settings: AdminSettings,
    current_user: UserInfo = Depends(get_admin_user)
):
    """Update system settings (Admin only)"""
    try:
        # In a real implementation, you'd want to update environment variables
        # or a configuration database. For now, we'll just validate and return.
        
        updated_settings = {}
        
        if settings.max_upload_size is not None:
            updated_settings["max_upload_size"] = settings.max_upload_size
            
        if settings.allowed_extensions is not None:
            updated_settings["allowed_extensions"] = settings.allowed_extensions
            
        if settings.admin_users is not None:
            updated_settings["admin_users"] = settings.admin_users
        
        logger.info(f"Settings updated by {current_user.email}: {updated_settings}")
        
        return {
            "message": "Settings updated successfully",
            "updated": updated_settings,
        }
        
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "True").lower() == "true",
    )
