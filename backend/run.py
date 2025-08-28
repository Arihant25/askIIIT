"""
Initializes the Jagruti backend API with database migrations
"""

import asyncio
import os
import logging
import uvicorn
from dotenv import load_dotenv
from pathlib import Path

# Configure logging
from colored_logging import setup_logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def main():
    """Start the API server"""
    try:        
        # Start the API server
        logger.info("Starting API server...")
        
        # Import and run the main app
        import main
        
        # Run the server
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))
        reload = os.getenv("DEBUG", "True").lower() == "true"
        
        logger.info(f"Starting server on {host}:{port} (reload={reload})")
        
        # Use uvicorn to run the server
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=reload,
        )
        
    except Exception as e:
        logger.error(f"Error initializing backend: {e}")
        raise

if __name__ == "__main__":
    # Run the main function with asyncio
    asyncio.run(main())
