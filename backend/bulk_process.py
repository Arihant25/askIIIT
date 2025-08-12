"""
Bulk document processing script with memory optimization for large document collections
"""

from colored_logging import setup_logging
from dotenv import load_dotenv
from document_processor import DocumentProcessor, DocumentSummarizer
import asyncio
import os
import sys
from pathlib import Path
import logging
import chromadb

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Import memory optimization utilities
try:
    from memory_config import setup_memory_optimized_environment
except ImportError:
    # Fallback if memory_config is not available
    def setup_memory_optimized_environment():
        return {}, None

# Load environment variables
load_dotenv()

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_existing_pdfs():
    """Process all PDFs in the pdfs directory with memory optimization"""
    
    # Setup memory optimization first
    memory_config, memory_monitor = setup_memory_optimized_environment()
    if memory_monitor:
        memory_monitor.log_memory_usage("before processing")

    # Initialize ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
        )

        # Create a proper ChromaDB-compatible embedding function using Qwen3
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        import torch

        # MEMORY OPTIMIZATION: Use appropriate device and settings
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

        # Use SentenceTransformerEmbeddingFunction which properly wraps the model
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            device=device,
            normalize_embeddings=True
        )

        # Create or get collections with the SentenceTransformer embedding function
        documents_collection = chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        chunks_collection = chroma_client.get_or_create_collection(
            name="chunks",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB initialized successfully")

        # Get already embedded filenames from ChromaDB
        try:
            existing_docs = documents_collection.get()
            embedded_filenames = set()
            for metadata in existing_docs.get("metadatas", []):
                name = metadata.get("name")
                if name:
                    embedded_filenames.add(name)
            logger.info(f"Found {len(embedded_filenames)} already embedded files in ChromaDB")
        except Exception as e:
            logger.warning(f"Could not fetch existing embedded files: {e}")
            embedded_filenames = set()

        if memory_monitor:
            memory_monitor.log_memory_usage("after ChromaDB initialization")

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        return

    # Initialize processors with memory optimization
    try:
        # ULTRA AGGRESSIVE: Force CPU mode for safety with 6GB GPU
        force_cpu = os.getenv("FORCE_CPU_EMBEDDINGS", "true").lower() == "true"
        logger.info(f"Initializing document processor (force_cpu={force_cpu})")
        
        # Pass memory config to document processor
        doc_processor = DocumentProcessor(memory_config=memory_config, force_cpu=force_cpu)
        summarizer = DocumentSummarizer()
        logger.info("Document processors initialized successfully with ultra-memory optimization")
        
        if memory_monitor:
            memory_monitor.log_memory_usage("after processor initialization")
            
    except Exception as e:
        logger.error(f"Failed to initialize processors: {e}")
        return

    # Find all PDF files
    pdfs_dir = backend_dir.parent / "pdfs"
    if not pdfs_dir.exists():
        logger.error(f"PDFs directory not found: {pdfs_dir}")
        return

    pdf_files = list(pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in pdfs directory")
        return

    # MEMORY OPTIMIZATION: Sort by file size and filter large files
    pdf_files_with_size = []
    max_file_size_mb = memory_config.get("max_file_size_mb", 25)  # Default 25MB limit
    
    for pdf_file in pdf_files:
        try:
            file_size = pdf_file.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size_mb > max_file_size_mb:
                logger.warning(f"Skipping {pdf_file.name} ({file_size_mb:.1f}MB) - exceeds size limit ({max_file_size_mb}MB)")
                continue
                
            pdf_files_with_size.append((pdf_file, file_size))
        except Exception as e:
            logger.warning(f"Could not get size for {pdf_file}: {e}")
            continue
    
    # Sort by file size (smallest first)
    pdf_files_with_size.sort(key=lambda x: x[1])
    pdf_files = [pdf_file for pdf_file, _ in pdf_files_with_size]

    logger.info(f"Found {len(pdf_files)} PDF files to process (sorted by size, max {max_file_size_mb}MB each)")

    # Process each PDF
    processed_count = 0
    failed_count = 0

    for i, pdf_file in enumerate(pdf_files):
        try:
            # Skip files that are already embedded
            if pdf_file.name in embedded_filenames:
                logger.info(f"Skipping {pdf_file.name} (already embedded)")
                continue

            # Check memory before each file
            if memory_monitor and not memory_monitor.check_memory_available(required_gb=1.0):
                logger.warning("Insufficient memory available, stopping processing")
                break

            file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
            logger.info(f"Processing {i+1}/{len(pdf_files)}: {pdf_file.name} ({file_size_mb:.1f} MB)")

            # MEMORY OPTIMIZATION: Read file in chunks for large files
            file_content = None
            try:
                with open(pdf_file, "rb") as f:
                    file_content = f.read()
            except MemoryError:
                logger.error(f"File {pdf_file.name} is too large to fit in memory")
                failed_count += 1
                continue

            # Generate document ID
            import uuid

            doc_id = str(uuid.uuid4())

            # Auto-categorize the document with memory optimization
            try:
                # MEMORY OPTIMIZATION: Use only a small sample for categorization
                sample_text = doc_processor.extract_text_from_file(
                    file_content[:100000], pdf_file.name  # Only first 100KB for sampling
                )
                category = summarizer.categorize_document(
                    pdf_file.name, sample_text[:1000]
                )
                description = summarizer.generate_summary(sample_text[:2000])
                logger.info(
                    f"Auto-categorized {pdf_file.name} as '{category}'")

                # Clear sample text from memory
                del sample_text
                import gc
                gc.collect()

            except Exception as e:
                logger.warning(
                    f"Could not auto-categorize {pdf_file.name}: {e}")
                category = "academics"  # Default category
                description = f"Document: {pdf_file.name}"

            # Process the document
            result = doc_processor.process_document(
                file_content=file_content,
                filename=pdf_file.name,
                doc_id=doc_id,
                category=category,
                description=description,
                author="system",
                documents_collection=documents_collection,
                chunks_collection=chunks_collection,
            )

            processed_count += 1
            logger.info(
                f"✓ Processed {pdf_file.name}: {result['chunk_count']} chunks created"
            )

            # MEMORY OPTIMIZATION: Clean up after each document
            del file_content
            if memory_monitor:
                memory_monitor.cleanup_memory()

            # Log memory usage periodically
            if memory_monitor and i % 2 == 0:  # Every 2 files
                memory_monitor.log_memory_usage(f"after processing {i+1} files")

        except Exception as e:
            failed_count += 1
            logger.error(f"✗ Failed to process {pdf_file.name}: {e}")

            # MEMORY OPTIMIZATION: Clean up on error
            if 'file_content' in locals():
                del file_content
            if memory_monitor:
                memory_monitor.cleanup_memory()

    logger.info(
        f"Bulk processing complete: {processed_count} processed, {failed_count} failed"
    )
    
    if memory_monitor:
        memory_monitor.log_memory_usage("after processing complete")


def check_existing_documents():
    """Check what documents are already in ChromaDB"""
    try:
        chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
        )

        try:
            documents_collection = chroma_client.get_collection(
                name="documents")
            doc_count = documents_collection.count()
            logger.info(f"Found {doc_count} existing documents in ChromaDB")

            if doc_count > 0:
                # Get some sample documents
                sample_docs = documents_collection.get(limit=5)
                logger.info("Sample documents:")
                for i, metadata in enumerate(sample_docs["metadatas"][:5]):
                    logger.info(
                        f"  {i+1}. {metadata.get('name', 'Unknown')} ({metadata.get('category', 'Unknown')})"
                    )

        except Exception:
            logger.info("No existing documents collection found")

        try:
            chunks_collection = chroma_client.get_collection(name="chunks")
            chunk_count = chunks_collection.count()
            logger.info(f"Found {chunk_count} existing chunks in ChromaDB")
        except Exception:
            logger.info("No existing chunks collection found")

    except Exception as e:
        logger.error(f"Error checking existing documents: {e}")


def reset_collections():
    """Reset ChromaDB collections to fix dimension mismatch"""
    try:
        chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
        )

        # Delete existing collections if they exist
        try:
            chroma_client.delete_collection(name="documents")
            logger.info("Deleted existing documents collection")
        except Exception:
            logger.info("No existing documents collection to delete")

        try:
            chroma_client.delete_collection(name="chunks")
            logger.info("Deleted existing chunks collection")
        except Exception:
            logger.info("No existing chunks collection to delete")

        # Create a proper ChromaDB-compatible embedding function using Qwen3
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        import torch

        # Use SentenceTransformerEmbeddingFunction which properly wraps the model
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            device="mps" if torch.backends.mps.is_available(
            ) else "cuda" if torch.cuda.is_available() else "cpu",
            normalize_embeddings=True
        )

        # Create new collections with correct settings
        documents_collection = chroma_client.create_collection(
            name="documents",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        chunks_collection = chroma_client.create_collection(
            name="chunks",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info("Created new collections with correct embedding function")

    except Exception as e:
        logger.error(f"Error resetting collections: {e}")
        raise


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Bulk process PDF documents")
    parser.add_argument(
        "--check", action="store_true", help="Check existing documents only"
    )
    parser.add_argument("--process", action="store_true",
                        help="Process all PDFs")
    parser.add_argument("--reset", action="store_true",
                        help="Reset ChromaDB collections to fix dimension issues")

    args = parser.parse_args()

    if args.check:
        check_existing_documents()
    elif args.process:
        asyncio.run(process_existing_pdfs())
    elif args.reset:
        reset_collections()
        logger.info(
            "Collections reset complete. You can now run --process to add documents.")
    else:
        print("Usage:")
        print("  python bulk_process.py --check     # Check existing documents")
        print("  python bulk_process.py --process   # Process all PDFs")
        print("  python bulk_process.py --reset     # Reset collections to fix dimension issues")


if __name__ == "__main__":
    main()
