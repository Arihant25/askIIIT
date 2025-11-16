"""
Setup script for Jagruti backend
This script helps with initial setup and model installation
"""

from colored_logging import setup_logging
import asyncio
import os
import sys
import subprocess
import logging
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))


setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_hf_tgi_server():
    """Check if HF TGI server is running"""
    try:
        from ollama_client import ollama_client
        return await ollama_client.check_connection()
    except Exception as e:
        logger.error(f"Failed to check HF TGI: {e}")
        return False


async def setup_inference_model():
    """Setup HF TGI inference model"""
    logger.info("Setting up HF TGI inference model...")

    # Check if HF TGI is running
    if not await check_hf_tgi_server():
        logger.error("Please start HF TGI Docker container first:")
        logger.error("docker run --gpus all \\")
        logger.error("    -v ~/.cache/huggingface:/root/.cache/huggingface \\")
        logger.error("    -p 8000:80 \\")
        logger.error("    ghcr.io/huggingface/text-generation-inference:latest \\")
        logger.error("    --model-id Qwen/Qwen3-1.7B")
        return False

    # Import client
    try:
        from ollama_client import ollama_client
    except ImportError:
        logger.error(
            "Could not import ollama_client. Make sure dependencies are installed."
        )
        return False

    # Setup chat model
    chat_model = ollama_client.chat_model
    logger.info(f"Using HF TGI chat model: {chat_model}")

    if not await ollama_client.ensure_model_available(chat_model):
        logger.error(f"Failed to verify chat model: {chat_model}")
        return False

    logger.info("✓ HF TGI inference model ready!")
    return True


async def test_embedding_model():
    """Test the Qwen-based embedding model"""
    logger.info("Testing Qwen-based embedding model...")

    try:
        from document_processor import DocumentProcessor

        # Initialize processor (this will download the model if needed)
        logger.info(
            "Initializing DocumentProcessor with Qwen-based embedding model...")
        doc_processor = DocumentProcessor()

        # Test embedding generation
        test_texts = ["Hello world", "This is a test document"]
        logger.info("Generating test embeddings...")
        embeddings = doc_processor.generate_embeddings(test_texts)

        logger.info(f"✓ Embedding model test successful!")
        logger.info(f"  - Generated embeddings for {len(test_texts)} texts")
        logger.info(f"  - Embedding dimension: {len(embeddings[0])}")

        return True

    except Exception as e:
        logger.error(f"Embedding model test failed: {e}")
        return False


async def setup_chromadb():
    """Setup ChromaDB"""
    logger.info("Setting up ChromaDB...")

    try:
        import chromadb
        from dotenv import load_dotenv

        load_dotenv()

        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(
            path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_data")
        )

        # Create collections
        documents_collection = chroma_client.get_or_create_collection(
            name="documents")
        chunks_collection = chroma_client.get_or_create_collection(
            name="chunks")

        logger.info("✓ ChromaDB setup complete!")
        logger.info(f"  - Documents: {documents_collection.count()}")
        logger.info(f"  - Chunks: {chunks_collection.count()}")

        return True

    except Exception as e:
        logger.error(f"ChromaDB setup failed: {e}")
        return False


def install_dependencies():
    """Install Python dependencies"""
    logger.info("Installing Python dependencies...")

    requirements_file = backend_dir / "requirements.txt"
    if not requirements_file.exists():
        logger.error("requirements.txt not found!")
        return False

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
                "-r", str(requirements_file)],
            check=True,
        )
        logger.info("✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


async def run_comprehensive_test():
    """Run a comprehensive test of the system"""
    logger.info("Running comprehensive system test...")

    try:
        # Test document processing pipeline
        from document_processor import DocumentProcessor, DocumentSummarizer

        doc_processor = DocumentProcessor()
        summarizer = DocumentSummarizer()

        # Test text extraction and processing
        test_text = "This is a test document for the Jagruti system. It contains information about IIIT Hyderabad."

        # Test chunking
        chunks = doc_processor.chunk_text(test_text)
        logger.info(f"✓ Text chunking: {len(chunks)} chunks created")

        # Test embeddings
        embeddings = doc_processor.generate_embeddings(chunks)
        logger.info(
            f"✓ Embedding generation: {len(embeddings)} embeddings created")

        # Test summarization
        summary = summarizer.generate_summary(test_text)
        logger.info(f"✓ Text summarization: '{summary[:50]}...'")

        # Test categorization
        category = summarizer.categorize_document("test.pdf", test_text)
        logger.info(f"✓ Document categorization: '{category}'")

        logger.info("✓ Comprehensive test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Comprehensive test failed: {e}")
        return False


async def main():
    """Main setup function"""
    logger.info("=== Jagruti Backend Setup ===")

    # Step 1: Install dependencies
    logger.info("\n1. Installing Python dependencies...")
    if not install_dependencies():
        logger.error("Setup failed at dependency installation")
        return False

    # Step 2: Setup ChromaDB
    logger.info("\n2. Setting up ChromaDB...")
    if not await setup_chromadb():
        logger.error("Setup failed at ChromaDB setup")
        return False

    # Step 3: Test embedding model
    logger.info("\n3. Testing Qwen-based embedding model...")
    if not await test_embedding_model():
        logger.error("Setup failed at embedding model test")
        return False

    # Step 4: Setup inference model
    logger.info("\n4. Setting up HF TGI inference model...")
    if not await setup_inference_model():
        logger.warning("HF TGI setup failed - chat functionality may not work")

    # Step 5: Run comprehensive test
    logger.info("\n5. Running comprehensive test...")
    if not await run_comprehensive_test():
        logger.error("Setup failed at comprehensive test")
        return False

    logger.info("\n=== Setup Complete! ===")
    logger.info("Your Jagruti backend is ready to use!")
    logger.info("\nNext steps:")
    logger.info("1. Start the API server: python run.py")
    logger.info("2. Process existing PDFs: python bulk_process.py --process")
    logger.info("3. Check API documentation: http://localhost:8000/docs")

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        sys.exit(1)
