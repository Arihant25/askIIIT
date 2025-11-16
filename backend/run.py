"""
Startup script for Jagruti backend
"""

from colored_logging import setup_logging
import asyncio
import os
import sys
import logging
from pathlib import Path

# Set PyTorch CUDA allocation config to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))


setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_environment():
    """Setup environment and check dependencies"""
    logger.info("Setting up Jagruti backend environment...")

    # Check if .env file exists
    env_file = backend_dir / ".env"
    if not env_file.exists():
        logger.warning(".env file not found. Using default configuration.")

    # Check ChromaDB data directory
    chroma_dir = backend_dir / "chroma_data"
    if not chroma_dir.exists():
        chroma_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created ChromaDB data directory: {chroma_dir}")

    # Check if pdfs directory exists
    pdfs_dir = backend_dir.parent / "pdfs"
    if pdfs_dir.exists() and any(pdfs_dir.glob("*.pdf")):
        logger.info(
            f"Found {len(list(pdfs_dir.glob('*.pdf')))} PDF files for processing"
        )
    else:
        logger.info("No PDF files found in pdfs directory")

    logger.info("Environment setup complete!")


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "chromadb",
        "pydantic",
        "httpx",
        "sentence-transformers",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(
            f"Missing required packages: {', '.join(missing_packages)}")
        logger.error(
            "Please install dependencies using: pip install -r requirements.txt"
        )
        return False

    logger.info("All required packages are installed!")
    return True


def main():
    """Main startup function"""
    logger.info("Starting Jagruti backend...")

    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    # Setup environment
    asyncio.run(setup_environment())

    # Start the FastAPI application
    import uvicorn
    from main import app

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    debug = os.getenv("DEBUG", "True").lower() == "true"

    logger.info(f"Starting server on {host}:{port} (debug={debug})")

    uvicorn.run("main:app", host=host, port=port,
                reload=debug, log_level="info")


if __name__ == "__main__":
    main()
