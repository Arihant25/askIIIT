"""
Forced migration script for the Jagruti RAG system.
This script ignores the migration flag and runs migrations again,
using the full document content for summarization and streaming the
summary generation in real-time.
"""

import os
import logging
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import time

# Ensure the backend directory is in the Python path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

# Import necessary modules
from colored_logging import setup_logging
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import torch

# Configure logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import the document summarizer and ollama client after env is loaded
from document_summarizer import DocumentSummaryProcessor
from ollama_client import ollama_client

async def stream_document_summary(doc_id: str, text: str, filename: str) -> str:
    """Generate a summary with streaming output for visual feedback"""
    try:
        logger.info(f"Generating streaming summary for document {doc_id} ({len(text)} chars)")
        
        # System prompt for summarization
        system_prompt = (
            "You are an academic document summarization assistant. "
            "Create a concise but comprehensive summary of the document. "
            "Include the key topics, main points, and important details. "
            "The summary should be informative enough to understand the document's content."
        )
        
        # Format prompt with document information
        prompt = (
            f"Summarize the following document:\n\n"
            f"Title: {filename}\n\n"
            f"Content: {text}\n\n"
            f"Summary:"
        )
        
        # Stream the summary generation
        full_summary = ""
        print(f"\n\n🔄 Generating summary for {filename} ({len(text)} chars)")
        print(f"📃 Summary streaming (thinking mode disabled): ", end="", flush=True)
        
        try:
            async for chunk in ollama_client.stream_response(
                prompt=prompt, 
                system_prompt=system_prompt
            ):
                # Print the chunk without newline
                print(chunk, end="", flush=True)
                full_summary += chunk
                # Add a short sleep to make the streaming visible
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Error streaming summary for document {doc_id}: {e}")
            print(f"\n❌ Error generating summary: {e}")
            # Return partial summary if we have any, otherwise create fallback
            if full_summary:
                return full_summary
            return f"Document: {filename}. (Summary generation failed)"
        
        print("\n✅ Summary generation complete!\n")
        return full_summary
        
    except Exception as e:
        logger.error(f"Error generating streaming summary for document {doc_id}: {e}")
        return f"Document: {filename}. (Summary generation failed)"

async def force_migrate():
    """
    Forcefully runs database migrations with full-text summarization and streaming
    """
    try:
        logger.info("Starting forced database migration for document summaries")
        
        # Check and remove migration flag if it exists
        migration_flag_file = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")) / "migration_completed"
        if migration_flag_file.exists():
            logger.info(f"Removing existing migration flag: {migration_flag_file}")
            migration_flag_file.unlink()
            logger.info("Migration flag removed, will perform complete migration")
        else:
            logger.info("No migration flag found, will perform fresh migration")
        
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
        )
        
        # Initialize embedding function
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            device="mps" if torch.backends.mps.is_available(
            ) else "cuda" if torch.cuda.is_available() else "cpu",
            normalize_embeddings=True
        )
        
        # Get existing collections
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
        
        # Check if summaries collection exists and drop it if it does
        try:
            existing_collections = chroma_client.list_collections()
            if any(col.name == "summaries" for col in existing_collections):
                logger.info("Dropping existing summaries collection")
                chroma_client.delete_collection(name="summaries")
                logger.info("Summaries collection dropped")
        except Exception as e:
            logger.warning(f"Error while checking/dropping summaries collection: {e}")
        
        # Create new summaries collection
        summaries_collection = chroma_client.create_collection(
            name="summaries",
            embedding_function=embedding_function,
            metadata={"description": "Document summaries with embeddings",
                    "hnsw:space": "cosine"}
        )
        
        logger.info("Created fresh summaries collection")
        
        # Get all documents with their full content
        logger.info("Fetching all documents for regenerating summaries")
        all_docs = documents_collection.get(
            include=["metadatas", "documents"],
            limit=10000
        )
        
        if all_docs and "ids" in all_docs and all_docs["ids"]:
            doc_count = len(all_docs["ids"])
            logger.info(f"Found {doc_count} documents to migrate")
            
            # Initialize document summary processor
            summary_processor = DocumentSummaryProcessor()
            
            # Process documents one by one with streaming summaries
            success_count = 0
            error_count = 0
            
            for i, doc_id in enumerate(all_docs["ids"]):
                try:
                    logger.info(f"\n\n======= Processing document {i+1}/{doc_count}: {doc_id} =======")
                    
                    # Get document data with full content
                    doc_results = documents_collection.get(
                        ids=[doc_id],
                        include=["documents", "metadatas"]
                    )
                    
                    if not doc_results["ids"]:
                        logger.warning(f"Document {doc_id} not found, skipping")
                        continue
                        
                    doc_metadata = doc_results["metadatas"][0]
                    doc_text = doc_results["documents"][0]
                    filename = doc_metadata.get("name", "unknown.pdf")
                    
                    # Log document size
                    logger.info(f"Document: {filename} - Size: {len(doc_text)} characters")
                    
                    # Extract categories from metadata or extract from content
                    categories = doc_metadata.get("categories", [])
                    if not categories:
                        single_category = doc_metadata.get("category")
                        if single_category:
                            categories = [single_category]
                        else:
                            # Extract categories using LLM with streaming feedback
                            print(f"🔍 Extracting categories for document: {filename}")
                            categories = await summary_processor.extract_document_categories(
                                doc_id, doc_text, filename
                            )
                            print(f"📁 Categories: {', '.join(categories)}")
                    
                    # Generate streaming summary using full document content
                    summary = await stream_document_summary(doc_id, doc_text, filename)
                    
                    # Generate embedding for the summary
                    print(f"🔢 Generating embedding for summary...")
                    summary_embedding = summary_processor.doc_processor.generate_embeddings([summary])[0]
                    print(f"✅ Embedding generated successfully")
                    
                    # Store in summaries collection
                    for category in categories:
                        summary_id = f"{doc_id}_{category}"
                        
                        summary_metadata = {
                            "doc_id": doc_id,
                            "summary_id": summary_id,
                            "category": category,  # Store single category, not list
                            "filename": filename,
                            "created_at": datetime.now().isoformat(),
                            "is_enhanced": True  # Mark as enhanced summary
                        }
                        
                        # Store in summaries collection
                        summaries_collection.add(
                            ids=[summary_id],
                            documents=[summary],
                            embeddings=[summary_embedding],
                            metadatas=[summary_metadata],
                        )
                    
                    # Update document metadata with category as string, not list
                    # ChromaDB expects simple types, not lists, for metadata values
                    updated_metadata = {**doc_metadata}
                    # Remove categories list if it exists
                    if "categories" in updated_metadata:
                        del updated_metadata["categories"]
                    # Store primary category as a string field
                    updated_metadata["category"] = categories[0] if categories else "academics"
                    # Store categories as comma-separated string
                    updated_metadata["category_list"] = ",".join(categories) if categories else "academics"
                    
                    documents_collection.update(
                        ids=[doc_id],
                        metadatas=[updated_metadata]
                    )
                    
                    success_count += 1
                    logger.info(f"✅ Successfully processed document {i+1}/{doc_count}: {filename}")
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"❌ Error processing document {doc_id}: {e}")
                    print(f"❌ Error processing document: {e}")
            
            logger.info(f"\n\n======= Migration Summary =======")
            logger.info(f"Total documents: {doc_count}")
            logger.info(f"Successfully processed: {success_count}")
            logger.info(f"Errors: {error_count}")
        else:
            logger.info("No documents found to migrate")
        
        # Create the migration flag file to prevent future automatic migrations
        migration_flag_file.parent.mkdir(parents=True, exist_ok=True)
        migration_flag_file.touch()
        logger.info(f"Created migration flag file: {migration_flag_file}")
        
        return {
            "status": "completed",
            "documents": documents_collection.count(),
            "chunks": chunks_collection.count(),
            "summaries": summaries_collection.count(),
            "successful": success_count,
            "errors": error_count
        }
        
    except Exception as e:
        logger.error(f"Error during forced migration: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Running forced database migration script")
        result = asyncio.run(force_migrate())
        logger.info(f"Migration result: {result}")
        logger.info("Migration completed successfully. You can now start the application normally.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)
