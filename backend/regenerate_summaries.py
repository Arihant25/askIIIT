"""
Selective document summary regeneration script for the Jagruti RAG system.
This script allows regenerating summaries for specific documents.
"""

import os
import logging
import asyncio
import sys
import argparse
from pathlib import Path

# Ensure the backend directory is in the Python path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

# Import necessary modules
from colored_logging import setup_logging
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from document_summarizer import DocumentSummaryProcessor
import torch

# Configure logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def regenerate_document_summary(doc_id=None, all_docs=False, list_docs=False, batch_size=5):
    """
    Regenerate summaries for specific documents or list available documents.
    
    Args:
        doc_id: Optional specific document ID to regenerate
        all_docs: If True, regenerate all document summaries
        list_docs: If True, just list available documents without regenerating
        batch_size: Batch size for processing multiple documents
    """
    try:
        logger.info("Initializing document summary regeneration")
        
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
        
        # Get summaries collection
        summaries_collection = chroma_client.get_or_create_collection(
            name="summaries",
            embedding_function=embedding_function,
            metadata={"description": "Document summaries with embeddings",
                    "hnsw:space": "cosine"}
        )
        
        # Get all documents
        all_docs_data = documents_collection.get(
            include=["metadatas"],
            limit=10000
        )
        
        if not all_docs_data or "ids" not in all_docs_data or not all_docs_data["ids"]:
            logger.warning("No documents found in the database")
            return {"status": "error", "message": "No documents found"}
            
        # Just list documents if requested
        if list_docs:
            doc_list = []
            for i, doc_id in enumerate(all_docs_data["ids"]):
                metadata = all_docs_data["metadatas"][i] if all_docs_data["metadatas"] else {}
                filename = metadata.get("name", "Unknown")
                categories = metadata.get("categories", [metadata.get("category", "unknown")])
                
                doc_list.append({
                    "id": doc_id,
                    "filename": filename,
                    "categories": categories
                })
                
                logger.info(f"Document {i+1}: {doc_id} - {filename} - Categories: {categories}")
            
            return {"status": "success", "documents": doc_list, "count": len(doc_list)}
        
        # Initialize document summary processor
        summary_processor = DocumentSummaryProcessor()
        
        # Process specific document if doc_id provided
        if doc_id:
            logger.info(f"Regenerating summary for document {doc_id}")
            
            # Get document data
            doc_results = documents_collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            if not doc_results["ids"]:
                logger.error(f"Document {doc_id} not found")
                return {"status": "error", "message": f"Document {doc_id} not found"}
            
            # Delete existing summaries for this document
            existing_summaries = summaries_collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"]
            )
            
            if existing_summaries["ids"]:
                logger.info(f"Deleting {len(existing_summaries['ids'])} existing summaries for document {doc_id}")
                summaries_collection.delete(ids=existing_summaries["ids"])
            
            # Get document data
            doc_metadata = doc_results["metadatas"][0]
            doc_text = doc_results["documents"][0]
            filename = doc_metadata.get("name", "unknown.pdf")
            
            # Extract categories
            categories = doc_metadata.get("categories", [])
            if not categories:
                single_category = doc_metadata.get("category")
                if single_category:
                    categories = [single_category]
                else:
                    categories = ["academics"]  # TODO: Set better default category
            
            # Process document summary
            try:
                result = summary_processor.process_document_summary(
                    doc_id=doc_id,
                    text=doc_text,
                    filename=filename,
                    categories=categories,
                    summaries_collection=summaries_collection
                )
                
                # Update document metadata with categories
                updated_metadata = {**doc_metadata, "categories": categories}
                documents_collection.update(
                    ids=[doc_id],
                    metadatas=[updated_metadata]
                )
                
                logger.info(f"Successfully regenerated summary for document {doc_id}")
                return {"status": "success", "document_id": doc_id, "categories": categories}
                
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {e}")
                return {"status": "error", "message": str(e), "document_id": doc_id}
        
        # Process all documents if requested
        elif all_docs:
            logger.info(f"Regenerating summaries for all {len(all_docs_data['ids'])} documents")
            
            result = summary_processor.regenerate_all_summaries(
                documents_collection=documents_collection,
                summaries_collection=summaries_collection,
                batch_size=batch_size
            )
            
            logger.info(f"All summaries regenerated: {result}")
            return {"status": "success", "result": result}
        
        else:
            logger.warning("No action specified (list_docs, doc_id, or all_docs required)")
            return {"status": "error", "message": "No action specified"}
        
    except Exception as e:
        logger.error(f"Error during summary regeneration: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate document summaries")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doc-id", help="Specific document ID to regenerate")
    group.add_argument("--all", action="store_true", help="Regenerate all document summaries")
    group.add_argument("--list", action="store_true", help="List all documents without regenerating")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing (default: 5)")
    
    args = parser.parse_args()
    
    try:
        if args.list:
            logger.info("Listing all documents")
            result = asyncio.run(regenerate_document_summary(list_docs=True))
        elif args.doc_id:
            logger.info(f"Regenerating summary for document {args.doc_id}")
            result = asyncio.run(regenerate_document_summary(doc_id=args.doc_id, batch_size=args.batch_size))
        elif args.all:
            logger.info("Regenerating summaries for all documents")
            result = asyncio.run(regenerate_document_summary(all_docs=True, batch_size=args.batch_size))
        
        logger.info(f"Operation completed: {result}")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)
