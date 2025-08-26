"""
Advanced Retrieval-Augmented Generation (RAG) pipeline for document search and retrieval
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import chromadb
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Implements a two-stage Retrieval-Augmented Generation pipeline:
    1. Stage 1: Retrieve relevant document summaries based on query
    2. Stage 2: Retrieve relevant content chunks from selected documents
    """
    
    def __init__(
        self,
        summaries_collection: chromadb.Collection,
        chunks_collection: chromadb.Collection,
        default_top_k: int = 5,
        summary_threshold: float = 0.15,  # Lowered default threshold
        content_threshold: float = 0.15   # Lowered default threshold
    ):
        """
        Initialize the RAG pipeline
        
        Args:
            summaries_collection: ChromaDB collection containing document summaries
            chunks_collection: ChromaDB collection containing document chunks
            default_top_k: Default number of top results to return if threshold not met
            summary_threshold: Similarity threshold for summary retrieval (lower value = more results)
            content_threshold: Similarity threshold for content retrieval (lower value = more results)
        """
        self.summaries_collection = summaries_collection
        self.chunks_collection = chunks_collection
        self.default_top_k = default_top_k
        self.summary_threshold = summary_threshold
        self.content_threshold = content_threshold
        
        # Configurable from environment
        self.summary_threshold = float(os.getenv("SUMMARY_THRESHOLD", str(summary_threshold)))
        self.content_threshold = float(os.getenv("CONTENT_THRESHOLD", str(content_threshold)))
        
        logger.info(f"RAG Pipeline initialized with thresholds: summary={self.summary_threshold}, content={self.content_threshold}")
    
    async def retrieve_relevant_documents(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        First stage of RAG: Retrieve relevant document summaries based on the query
        
        Returns:
            Tuple containing:
            - List of relevant document metadata
            - List of unique document IDs that matched the query
        """
        try:
            # Default values
            top_k = top_k or self.default_top_k
            threshold = threshold or self.summary_threshold
            
            # Build query filter based on categories
            where_clause = {}
            if categories and len(categories) > 0:
                if len(categories) == 1:
                    where_clause["category"] = categories[0]
                else:
                    where_clause["category"] = {"$in": categories}
            
            logger.info(f"Querying summaries with where clause: {where_clause}")
            
            # Query summaries collection
            results = self.summaries_collection.query(
                query_texts=[query],
                n_results=min(50, top_k * 3),  # Get more results than needed for filtering
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]  # Explicitly specify what to include
            )
            
            # Process and filter results
            relevant_docs = []
            unique_doc_ids = set()
            
            if results["ids"] and len(results["ids"]) > 0 and len(results["ids"][0]) > 0:
                # First, collect all results
                all_results = []
                
                for i, (summary_id, summary, metadata, distance) in enumerate(zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Calculate relevance score
                    relevance_score = 1.0 - distance
                    
                    # Include all results, even with low relevance
                    doc_id = metadata.get("doc_id")
                    
                    result_item = {
                        "summary_id": summary_id,
                        "doc_id": doc_id,
                        "category": metadata.get("category"),
                        "filename": metadata.get("filename", "Unknown"),
                        "summary": summary,
                        "relevance_score": round(relevance_score, 3),
                        "distance": round(distance, 3),
                    }
                    
                    all_results.append(result_item)
                
                # Sort by relevance score (highest first)
                all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
                
                # Always take top K results, apply threshold only if we have plenty of results
                if len(all_results) > top_k * 2:
                    # Apply threshold only if we have plenty of results
                    above_threshold = [r for r in all_results if r["relevance_score"] >= threshold]
                    if len(above_threshold) >= top_k:
                        relevant_docs = above_threshold[:top_k]
                    else:
                        relevant_docs = all_results[:top_k]
                else:
                    # Not enough results to be picky, just take top K
                    relevant_docs = all_results[:top_k] if len(all_results) >= top_k else all_results
                
                # Extract unique document IDs
                unique_doc_ids = {doc["doc_id"] for doc in relevant_docs if "doc_id" in doc}
            
            logger.info(f"Retrieved {len(relevant_docs)} relevant documents ({len(unique_doc_ids)} unique) for query: '{query}'")
            
            return relevant_docs, list(unique_doc_ids)
            
        except Exception as e:
            logger.error(f"Error retrieving relevant documents: {e}")
            return [], []
    
    async def retrieve_content_chunks(
        self,
        query: str,
        doc_ids: List[str],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Second stage of RAG: Retrieve relevant content chunks from the selected documents
        """
        try:
            if not doc_ids:
                logger.warning("No document IDs provided for content retrieval")
                return []
            
            # Default values
            top_k = top_k or self.default_top_k
            threshold = threshold or self.content_threshold
            
            logger.info(f"Retrieving content chunks for {len(doc_ids)} documents with query: '{query}'")
            
            # Query chunks collection with document filter
            results = self.chunks_collection.query(
                query_texts=[query],
                n_results=min(100, top_k * 5),  # Get more results for filtering
                where={"doc_id": {"$in": doc_ids}},
                include=["documents", "metadatas", "distances"]  # Explicitly specify what to include
            )
            
            # Process and filter results
            relevant_chunks = []
            
            if results["ids"] and len(results["ids"]) > 0 and len(results["ids"][0]) > 0:
                # Collect all results
                all_chunks = []
                
                for i, (chunk_id, chunk_text, metadata, distance) in enumerate(zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Calculate relevance score
                    relevance_score = 1.0 - distance
                    
                    # Include all results for filtering
                    chunk_item = {
                        "chunk_id": chunk_id,
                        "doc_id": metadata.get("doc_id"),
                        "category": metadata.get("category"),
                        "filename": metadata.get("filename", "Unknown"),
                        "text": chunk_text,
                        "relevance_score": round(relevance_score, 3),
                        "distance": round(distance, 3),
                    }
                    
                    all_chunks.append(chunk_item)
                
                # Sort by relevance score (highest first)
                all_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
                
                # Always take top K chunks, apply threshold only if we have plenty of results
                if len(all_chunks) > top_k * 2:
                    # Apply threshold only if we have plenty of results
                    above_threshold = [c for c in all_chunks if c["relevance_score"] >= threshold]
                    if len(above_threshold) >= top_k:
                        relevant_chunks = above_threshold[:top_k]
                    else:
                        relevant_chunks = all_chunks[:top_k]
                else:
                    # Not enough results to be picky, just take top K
                    relevant_chunks = all_chunks[:top_k] if len(all_chunks) >= top_k else all_chunks
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant content chunks for query across {len(doc_ids)} documents")
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving content chunks: {e}")
            return []
    
    async def retrieve(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        include_summaries: bool = True
    ) -> Dict[str, Any]:
        """
        Complete two-stage RAG retrieval pipeline
        
        Args:
            query: The user's query
            categories: Optional list of categories to filter by
            top_k: Number of results to return (default: self.default_top_k)
            include_summaries: Whether to include document summaries in results
            
        Returns:
            Dictionary with retrieved documents and content
        """
        try:
            logger.info(f"Starting RAG retrieval for query: '{query}', categories: {categories}")
            
            # Stage 1: Retrieve relevant documents based on summaries
            relevant_docs, doc_ids = await self.retrieve_relevant_documents(
                query=query,
                categories=categories,
                top_k=top_k or self.default_top_k
            )
            
            if not doc_ids:
                # Log collection info for debugging
                summaries_count = self.summaries_collection.count()
                chunks_count = self.chunks_collection.count()
                logger.warning(f"No relevant documents found for query: '{query}'. "
                              f"Collection stats - Summaries: {summaries_count}, Chunks: {chunks_count}")
                
                # Try to get a sample of available summaries for debugging
                try:
                    sample_summaries = self.summaries_collection.get(
                        limit=3,
                        include=["metadatas"]
                    )
                    
                    if sample_summaries["ids"] and len(sample_summaries["ids"]) > 0:
                        sample_categories = [m.get("category", "unknown") for m in sample_summaries["metadatas"]]
                        logger.info(f"Sample available summaries - IDs: {sample_summaries['ids']}, "
                                   f"Categories: {sample_categories}")
                except Exception as e:
                    logger.error(f"Error sampling summaries: {e}")
                
                return {
                    "query": query,
                    "documents": [],
                    "chunks": [],
                    "total_docs": 0,
                    "total_chunks": 0,
                }
            
            # Stage 2: Retrieve relevant content chunks from those documents
            relevant_chunks = await self.retrieve_content_chunks(
                query=query,
                doc_ids=doc_ids,
                top_k=top_k or self.default_top_k
            )
            
            # Only include document summaries if requested
            if not include_summaries:
                relevant_docs = []
            
            return {
                "query": query,
                "documents": relevant_docs,
                "chunks": relevant_chunks,
                "total_docs": len(relevant_docs),
                "total_chunks": len(relevant_chunks),
            }
            
        except Exception as e:
            logger.error(f"Error in RAG retrieval pipeline: {e}")
            return {
                "query": query,
                "documents": [],
                "chunks": [],
                "total_docs": 0,
                "total_chunks": 0,
                "error": str(e)
            }
    
    def process_conversation_context(
        self,
        current_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_history: int = 5
    ) -> str:
        """
        Process conversation context to enhance the query with recent conversation history
        
        Args:
            current_message: The current user message
            conversation_history: List of previous messages
            max_history: Maximum number of previous messages to include
            
        Returns:
            Enhanced query with conversation context
        """
        try:
            if not conversation_history or len(conversation_history) == 0:
                return current_message
            
            # Take only recent messages up to max_history
            recent_history = conversation_history[-max_history:] if len(conversation_history) > max_history else conversation_history
            
            # Build context string
            context = "Context from previous conversation:\n"
            
            for msg in recent_history:
                role = msg.get("type", "unknown")
                content = msg.get("content", "")
                
                if role and content:
                    role_display = "User" if role == "user" else "Assistant"
                    context += f"{role_display}: {content}\n"
            
            # Combine context with current message
            enhanced_query = f"{context}\n\nCurrent question: {current_message}"
            
            logger.debug(f"Enhanced query with {len(recent_history)} conversation messages")
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error processing conversation context: {e}")
            return current_message  # Fallback to original message
