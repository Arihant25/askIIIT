"""
Bulk document processing s    # Initialize ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
        )
        
        # Create or get collections with manual embeddings (disable default embedding function)
        # Use cosine similarity for better semantic search
        documents_collection = chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=None,  # Disable auto-embedding since we provide our own
            metadata={"hnsw:space": "cosine"}
        )
        chunks_collection = chroma_client.get_or_create_collection(
            name="chunks", 
            embedding_function=None,  # Disable auto-embedding since we provide our own
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        returnDFs
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


# Load environment variables
load_dotenv()


setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_existing_pdfs():
    """Process all PDFs in the pdfs directory"""

    # Initialize ChromaDB
    try:
        chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
        )
        
        # Custom embedding function for ChromaDB
        class CustomEmbeddingFunction:
            def __init__(self):
                from document_processor import DocumentProcessor
                self.doc_processor = DocumentProcessor()
            
            def __call__(self, input):
                if isinstance(input, str):
                    input = [input]
                embeddings = self.doc_processor.generate_embeddings(input)
                return embeddings

        # Initialize embedding function
        embedding_function = CustomEmbeddingFunction()
        
        # Create or get collections with the custom embedding function
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
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        return

    # Initialize processors
    try:
        doc_processor = DocumentProcessor()
        summarizer = DocumentSummarizer()
        logger.info("Document processors initialized successfully")
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

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Process each PDF
    processed_count = 0
    failed_count = 0

    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")

            # Read file content
            with open(pdf_file, "rb") as f:
                file_content = f.read()

            # Generate document ID
            import uuid

            doc_id = str(uuid.uuid4())

            # Auto-categorize the document
            try:
                sample_text = doc_processor.extract_text_from_file(
                    file_content, pdf_file.name
                )
                category = summarizer.categorize_document(
                    pdf_file.name, sample_text[:1000]
                )
                description = summarizer.generate_summary(sample_text[:2000])
                logger.info(
                    f"Auto-categorized {pdf_file.name} as '{category}'")
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

        except Exception as e:
            failed_count += 1
            logger.error(f"✗ Failed to process {pdf_file.name}: {e}")

    logger.info(
        f"Bulk processing complete: {processed_count} processed, {failed_count} failed"
    )


def check_existing_documents():
    """Check what documents are already in ChromaDB"""
    try:
        chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
        )

        try:
            # Custom embedding function for ChromaDB
            class CustomEmbeddingFunction:
                def __init__(self):
                    from document_processor import DocumentProcessor
                    self.doc_processor = DocumentProcessor()
                
                def __call__(self, input):
                    if isinstance(input, str):
                        input = [input]
                    embeddings = self.doc_processor.generate_embeddings(input)
                    return embeddings

            # Initialize embedding function
            embedding_function = CustomEmbeddingFunction()
            
            documents_collection = chroma_client.get_collection(name="documents")
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
            
        # Custom embedding function for ChromaDB
        class CustomEmbeddingFunction:
            def __init__(self):
                from document_processor import DocumentProcessor
                self.doc_processor = DocumentProcessor()
            
            def __call__(self, input):
                if isinstance(input, str):
                    input = [input]
                embeddings = self.doc_processor.generate_embeddings(input)
                return embeddings

        # Initialize embedding function
        embedding_function = CustomEmbeddingFunction()
            
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
    parser.add_argument("--process", action="store_true", help="Process all PDFs")
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB collections to fix dimension issues")

    args = parser.parse_args()

    if args.check:
        check_existing_documents()
    elif args.process:
        asyncio.run(process_existing_pdfs())
    elif args.reset:
        reset_collections()
        logger.info("Collections reset complete. You can now run --process to add documents.")
    else:
        print("Usage:")
        print("  python bulk_process.py --check     # Check existing documents")
        print("  python bulk_process.py --process   # Process all PDFs")
        print("  python bulk_process.py --reset     # Reset collections to fix dimension issues")


if __name__ == "__main__":
    main()
