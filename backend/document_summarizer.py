"""
Document summarization pipeline for creating and managing document summaries
"""

import os
import logging
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Import necessary modules
from document_processor import DocumentProcessor
from ollama_client import ollama_client

logger = logging.getLogger(__name__)

class DocumentSummaryProcessor:
    """
    Handles creating and storing document summaries with embeddings 
    for faster retrieval in the RAG pipeline
    """
    
    def __init__(self, embedding_model_name: Optional[str] = None):
        """Initialize the document summary processor"""
        try:
            model_name = embedding_model_name or os.getenv(
                "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
            )
            
            # Initialize the document processor for embeddings
            self.doc_processor = DocumentProcessor(embedding_model_name=model_name)
            
            # Thread-safe locks
            self._lock = threading.RLock()
            
            logger.info(f"Document summary processor initialized with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize document summary processor: {e}")
            raise
    
    def _run_async(self, coroutine, timeout=600):
        """Helper method to safely run async coroutines from sync code"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, use run_coroutine_threadsafe
                logger.info(f"Running async operation with {timeout}s timeout")
                future = asyncio.run_coroutine_threadsafe(coroutine, loop)
                return future.result(timeout=timeout)
            else:
                # If no event loop is running, use asyncio.run
                return asyncio.run(coroutine)
        except asyncio.TimeoutError:
            logger.error(f"Timeout while running async coroutine (exceeded {timeout}s)")
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Error running async coroutine: {e}")
            raise
    
    async def generate_document_summary(self, doc_id: str, text: str, filename: str) -> str:
        """Generate a concise summary of the document text using LLM"""
        try:
            # Log document size for monitoring
            logger.info(f"Generating summary for document {doc_id} ({filename}) with {len(text)} characters")
            
            # Add timeout handling for LLM call with full document text
            try:
                # Use a much longer timeout for larger documents
                timeout_seconds = max(300, len(text) // 5000 * 60)  # Base 5 min + 1 min per 5000 chars
                logger.info(f"Using {timeout_seconds}s timeout for document {doc_id} summarization")
                
                summary = await asyncio.wait_for(
                    ollama_client.summarize_text(text, max_length=500),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning(f"LLM summarization timed out after {timeout_seconds}s for document {doc_id}")
                # Create a more informative fallback summary
                return f"Document: {filename}. This is a large document containing {len(text)} characters covering topics related to IIIT Hyderabad."
            
            logger.info(f"Successfully generated summary for document {doc_id} ({len(summary)} chars)")
            return summary
        except Exception as e:
            logger.error(f"Error generating summary for document {doc_id}: {e}")
            # Fallback to basic summary
            return f"Document: {filename}. Content summary not available due to processing error."
    
    async def extract_document_categories(self, doc_id: str, text: str, filename: str) -> List[str]:
        """Extract multiple categories from document content using LLM"""
        try:
            # Take beginning, middle, and end samples for better categorization
            # without truncating important content
            text_length = len(text)
            logger.info(f"Extracting categories for document {doc_id} ({filename}) with {text_length} characters")
            
            # For very large documents, sample strategically
            if text_length > 6000:
                # Take beginning (first 2000 chars)
                beginning = text[:2000]
                
                # Take middle (1000 chars from middle)
                middle_start = text_length // 2 - 500
                middle = text[middle_start:middle_start + 1000]
                
                # Take end (last 1000 chars)
                end = text[-1000:]
                
                sampled_text = f"{beginning}\n\n[...middle portion omitted...]\n\n{middle}\n\n[...]\n\n{end}"
                logger.info(f"Created strategic sample from document {doc_id} (beginning+middle+end: 4000 chars)")
            else:
                # For smaller documents, use the full content
                sampled_text = text
            
            system_prompt = (
                "You are a document categorization assistant. "
                "Categorize the following document into one or more of these categories: "
                "faculty, student, hostel, academics, mess. "
                "A document can belong to multiple categories if relevant. "
                "Respond with only the category names separated by commas, nothing else."
            )

            prompt = (
                f"Filename: {filename}\n\n"
                f"Content: {sampled_text}\n\n"
                "Categories (comma-separated):"
            )

            # Add timeout handling for LLM call
            try:
                category_text = await asyncio.wait_for(
                    ollama_client.generate_response(prompt=prompt, system_prompt=system_prompt),
                    timeout=120  # 2-minute timeout for categorization
                )
            except asyncio.TimeoutError:
                logger.warning(f"LLM categorization timed out for document {doc_id}, using default category")
                return ["academics"]  # Default category on timeout
            
            # Parse categories
            categories = [
                cat.strip().lower() 
                for cat in category_text.split(",") 
                if cat.strip()
            ]
            
            # Filter to valid categories
            valid_categories = {"faculty", "student", "hostel", "academics", "mess"}
            categories = [cat for cat in categories if cat in valid_categories]
            
            # Add at least one category if none were found
            if not categories:
                # Default to "academics" if no categories detected
                categories = ["academics"]
            
            logger.info(f"Extracted categories for document {doc_id}: {categories}")
            return categories
            
        except Exception as e:
            logger.error(f"Error extracting categories for document {doc_id}: {e}")
            return ["academics"]  # Default fallback
    
    def process_document_summary(
        self,
        doc_id: str,
        text: str,
        filename: str,
        categories: List[str],
        summaries_collection: chromadb.Collection
    ) -> Dict[str, Any]:
        """
        Process document summary and store in the summaries collection
        """
        try:
            logger.info(f"Processing summary for document {doc_id} in {len(categories)} categories")
            
            # Generate summary using LLM (safely handle async)
            summary = self._run_async(self.generate_document_summary(doc_id, text, filename))
            
            # Generate embedding for the summary
            summary_embedding = self.doc_processor.generate_embeddings([summary])[0]
            
            # Create an entry for each category
            for category in categories:
                summary_id = f"{doc_id}_{category}"
                
                summary_metadata = {
                    "doc_id": doc_id,
                    "summary_id": summary_id,
                    "category": category,
                    "filename": filename,
                    "created_at": datetime.now().isoformat(),
                }
                
                # Store in summaries collection
                summaries_collection.add(
                    ids=[summary_id],
                    documents=[summary],
                    embeddings=[summary_embedding],
                    metadatas=[summary_metadata],
                )
            
            logger.info(f"Successfully processed summary for document {doc_id} in {len(categories)} categories")
            
            return {
                "doc_id": doc_id,
                "summary": summary,
                "categories": categories,
                "status": "completed",
            }
            
        except Exception as e:
            logger.error(f"Error processing summary for document {doc_id}: {e}")
            raise
    
    def update_document_categories(
        self,
        doc_id: str,
        categories: List[str],
        summaries_collection: chromadb.Collection,
        documents_collection: chromadb.Collection
    ) -> Dict[str, Any]:
        """
        Update document categories by adding new category entries or removing old ones
        """
        try:
            logger.info(f"Updating categories for document {doc_id}: {categories}")
            
            # Get existing document data
            doc_results = documents_collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            if not doc_results["ids"]:
                raise ValueError(f"Document {doc_id} not found")
            
            doc_metadata = doc_results["metadatas"][0]
            doc_text = doc_results["documents"][0]
            filename = doc_metadata.get("name", "unknown.pdf")
            
            # Get existing summaries for this document
            existing_summaries = summaries_collection.get(
                where={"doc_id": doc_id},
                include=["metadatas", "documents", "embeddings"]
            )
            
            existing_categories = set()
            if existing_summaries["ids"]:
                existing_categories = {
                    meta.get("category") for meta in existing_summaries["metadatas"]
                }
            
            # Categories to add (new ones)
            categories_to_add = set(categories) - existing_categories
            
            # Categories to remove (old ones not in new list)
            categories_to_remove = existing_categories - set(categories)
            
            # Remove old categories
            for category in categories_to_remove:
                summary_id = f"{doc_id}_{category}"
                summaries_collection.delete(ids=[summary_id])
                logger.info(f"Removed category {category} for document {doc_id}")
            
            # Add new categories
            if categories_to_add:
                # If we have existing summary data, reuse it
                if existing_summaries["documents"] and existing_summaries["embeddings"]:
                    summary = existing_summaries["documents"][0]
                    summary_embedding = existing_summaries["embeddings"][0]
                    
                    for category in categories_to_add:
                        summary_id = f"{doc_id}_{category}"
                        
                        summary_metadata = {
                            "doc_id": doc_id,
                            "summary_id": summary_id,
                            "category": category,
                            "filename": filename,
                            "created_at": datetime.now().isoformat(),
                        }
                        
                        # Store in summaries collection
                        summaries_collection.add(
                            ids=[summary_id],
                            documents=[summary],
                            embeddings=[summary_embedding],
                            metadatas=[summary_metadata],
                        )
                        
                        logger.info(f"Added new category {category} for document {doc_id}")
                
                # If no existing summary, process the document text
                else:
                    # Generate a new summary and process it
                    self.process_document_summary(
                        doc_id=doc_id,
                        text=doc_text,
                        filename=filename,
                        categories=list(categories_to_add),
                        summaries_collection=summaries_collection
                    )
            
            # Update document metadata with all categories
            updated_metadata = {**doc_metadata, "categories": categories}
            documents_collection.update(
                ids=[doc_id],
                metadatas=[updated_metadata]
            )
            
            return {
                "doc_id": doc_id,
                "categories_added": list(categories_to_add),
                "categories_removed": list(categories_to_remove),
                "current_categories": categories,
                "status": "completed",
            }
            
        except Exception as e:
            logger.error(f"Error updating categories for document {doc_id}: {e}")
            raise
    
    def regenerate_all_summaries(
        self,
        documents_collection: chromadb.Collection,
        summaries_collection: chromadb.Collection,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Regenerate summaries for all documents in the database
        """
        try:
            logger.info("Starting summary regeneration for all documents")
            
            # Get all documents
            all_docs = documents_collection.get(
                include=["metadatas"],
                limit=10000  # Adjust as needed
            )
            
            if not all_docs["ids"]:
                return {"status": "completed", "processed": 0, "message": "No documents found"}
            
            doc_count = len(all_docs["ids"])
            logger.info(f"Found {doc_count} documents for summary regeneration")
            
            # Process in batches
            success_count = 0
            error_count = 0
            skipped_count = 0
            
            for i in range(0, doc_count, batch_size):
                batch_end = min(i + batch_size, doc_count)
                batch_docs = all_docs["ids"][i:batch_end]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(doc_count + batch_size - 1)//batch_size}")
                
                for doc_id in batch_docs:
                    try:
                        # Get full document data
                        doc_results = documents_collection.get(
                            ids=[doc_id],
                            include=["documents", "metadatas"]
                        )
                        
                        if not doc_results["ids"]:
                            logger.warning(f"Document {doc_id} not found, skipping")
                            skipped_count += 1
                            continue
                        
                        doc_metadata = doc_results["metadatas"][0]
                        doc_text = doc_results["documents"][0]
                        filename = doc_metadata.get("name", "unknown.pdf")
                        
                        # Check if document text is too short
                        if not doc_text or len(doc_text.strip()) < 50:
                            logger.warning(f"Document {doc_id} has insufficient text ({len(doc_text)} chars), skipping")
                            skipped_count += 1
                            continue
                        
                        # Extract categories
                        categories = doc_metadata.get("categories", [])
                        
                        # If no categories field, extract from document
                        if not categories:
                            # Try to get from "category" field first
                            single_category = doc_metadata.get("category")
                            if single_category:
                                categories = [single_category]
                            else:
                                # Extract categories using LLM (safely handle async)
                                try:
                                    categories = self._run_async(
                                        self.extract_document_categories(doc_id, doc_text, filename),
                                        timeout=90  # 1.5 minutes timeout
                                    )
                                except Exception as cat_error:
                                    logger.error(f"Error extracting categories for document {doc_id}: {cat_error}")
                                    categories = ["academics"]  # Default category on error
                        
                        # Delete existing summaries for this document
                        try:
                            existing_summaries = summaries_collection.get(
                                where={"doc_id": doc_id},
                                include=["metadatas"]
                            )
                            
                            if existing_summaries["ids"]:
                                summaries_collection.delete(ids=existing_summaries["ids"])
                                logger.info(f"Deleted {len(existing_summaries['ids'])} existing summaries for document {doc_id}")
                        except Exception as del_error:
                            logger.error(f"Error deleting existing summaries for document {doc_id}: {del_error}")
                            # Continue anyway to create new summaries
                        
                        # Process and store new summary
                        try:
                            # Log document size
                            doc_size = len(doc_text)
                            logger.info(f"Processing document {doc_id} ({filename}) - Size: {doc_size} characters")
                            
                            # Track start time for performance monitoring
                            start_time = datetime.now()
                            
                            # Process document
                            self.process_document_summary(
                                doc_id=doc_id,
                                text=doc_text,
                                filename=filename,
                                categories=categories,
                                summaries_collection=summaries_collection
                            )
                            
                            # Calculate processing time
                            processing_time = (datetime.now() - start_time).total_seconds()
                            logger.info(f"Successfully processed document {doc_id} in {processing_time:.2f} seconds")
                            
                            # Update document metadata with categories
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
                            logger.info(f"Successfully regenerated summary for document {doc_id}")
                            
                        except Exception as summary_error:
                            error_count += 1
                            logger.error(f"Error processing summary for document {doc_id}: {summary_error}")
                            
                            # Try to create a basic fallback summary
                            try:
                                logger.info(f"Creating fallback summary for document {doc_id}")
                                # Create a basic summary from the start of the document
                                basic_summary = f"Document: {filename}. " + doc_text[:500].replace("\n", " ")
                                
                                # Generate embedding for the basic summary
                                summary_embedding = self.doc_processor.generate_embeddings([basic_summary])[0]
                                
                                # Store in summaries collection
                                for category in categories:
                                    summary_id = f"{doc_id}_{category}"
                                    
                                    summary_metadata = {
                                        "doc_id": doc_id,
                                        "summary_id": summary_id,
                                        "category": category,
                                        "filename": filename,
                                        "created_at": datetime.now().isoformat(),
                                        "is_fallback": True
                                    }
                                    
                                    # Store in summaries collection
                                    summaries_collection.add(
                                        ids=[summary_id],
                                        documents=[basic_summary],
                                        embeddings=[summary_embedding],
                                        metadatas=[summary_metadata],
                                    )
                                
                                # Update document metadata with categories
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
                                
                                logger.info(f"Created fallback summary for document {doc_id}")
                                # Count as success since we created a fallback
                                success_count += 1
                                error_count -= 1  # Remove from error count since we recovered
                            except Exception as fallback_error:
                                logger.error(f"Failed to create fallback summary for document {doc_id}: {fallback_error}")
                        
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error regenerating summary for document {doc_id}: {e}")
                
                # Log progress after each batch
                logger.info(f"Batch progress: {success_count} successful, {error_count} errors, {skipped_count} skipped")
            
            logger.info(f"Summary regeneration completed: {success_count} successful, {error_count} errors, {skipped_count} skipped")
            
            return {
                "status": "completed",
                "total": doc_count,
                "successful": success_count,
                "errors": error_count,
                "skipped": skipped_count,
            }
            
        except Exception as e:
            logger.error(f"Error in summary regeneration: {e}")
            raise
