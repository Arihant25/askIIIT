"""
Document processing utilities for extracting text and creating embeddings
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import pdfplumber
import io
from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, embedding_model_name: Optional[str] = None):
        """Initialize document processor with Qwen3 embedding model"""
        try:
            model_name = embedding_model_name or os.getenv(
                "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"
            )
            device = os.getenv("EMBEDDING_DEVICE", "auto")
            trust_remote_code = (
                os.getenv("EMBEDDING_TRUST_REMOTE_CODE", "true").lower() == "true"
            )

            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(
                model_name, device=device, trust_remote_code=trust_remote_code
            )
            logger.info(f"Successfully loaded Qwen-based embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file content using pdfplumber"""
        try:
            pdf_file = io.BytesIO(file_content)
            text = ""

            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        """Extract text from various file types"""
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension == ".pdf":
            return self.extract_text_from_pdf(file_content)
        elif file_extension in [".txt", ".md"]:
            return file_content.decode("utf-8", errors="ignore")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)

            # Stop if we've reached the end
            if i + chunk_size >= len(words):
                break

        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using Qwen3-Embedding-8B"""
        try:
            logger.info(
                f"Generating embeddings for {len(texts)} texts using Qwen3-Embedding-8B"
            )

            # Generate embeddings using sentence-transformers
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=len(texts) > 10,
                batch_size=32,  # Adjust based on your GPU memory
                normalize_embeddings=True,  # Normalize for better similarity search
            )

            # Convert to list if numpy array
            if hasattr(embeddings, "tolist"):
                embeddings = embeddings.tolist()

            logger.info(
                f"Successfully generated embeddings with dimension: {len(embeddings[0]) if embeddings else 0}"
            )
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def process_document(
        self,
        file_content: bytes,
        filename: str,
        doc_id: str,
        category: str,
        description: str,
        author: str,
        documents_collection: chromadb.Collection,
        chunks_collection: chromadb.Collection,
    ) -> Dict[str, Any]:
        """Process a complete document: extract text, create chunks, generate embeddings, store in ChromaDB"""
        try:
            # Extract text
            text = self.extract_text_from_file(file_content, filename)

            if not text.strip():
                raise ValueError("No text content found in document")

            # Create chunks
            chunks = self.chunk_text(text)

            if not chunks:
                raise ValueError("No chunks created from document")

            # Generate embeddings
            embeddings = self.generate_embeddings(chunks)

            # Prepare chunk data for ChromaDB
            chunk_ids = []
            chunk_texts = []
            chunk_embeddings = []
            chunk_metadatas = []

            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = str(uuid.uuid4())
                chunk_ids.append(chunk_id)
                chunk_texts.append(chunk_text)
                chunk_embeddings.append(embedding)

                chunk_metadata = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "position": i,
                    "category": category,
                    "filename": filename,
                    "author": author,
                    "created_at": datetime.now().isoformat(),
                }
                chunk_metadatas.append(chunk_metadata)

            # Store chunks in ChromaDB
            chunks_collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                embeddings=chunk_embeddings,
                metadatas=chunk_metadatas,
            )

            # Store document metadata
            doc_metadata = {
                "doc_id": doc_id,
                "name": filename,
                "category": category,
                "description": description,
                "author": author,
                "chunk_count": len(chunks),
                "text_length": len(text),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }

            documents_collection.add(
                ids=[doc_id],
                documents=[text[:1000]],  # Store first 1000 chars as summary
                metadatas=[doc_metadata],
            )

            logger.info(
                f"Successfully processed document {filename}: {len(chunks)} chunks created"
            )

            return {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_count": len(chunks),
                "text_length": len(text),
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            raise


class DocumentSummarizer:
    """Generate summaries and metadata for documents using Qwen3 via Ollama"""

    def __init__(self):
        try:
            from ollama_client import ollama_client

            self.ollama_client = ollama_client
            logger.info("Initialized document summarizer with Ollama client")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client for summarization: {e}")
            self.ollama_client = None

    async def generate_summary_async(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the document text using Qwen3"""
        if not self.ollama_client:
            return self._fallback_summary(text, max_length)

        try:
            summary = await self.ollama_client.summarize_text(text, max_length)
            return summary
        except Exception as e:
            logger.error(f"Error generating summary with Ollama: {e}")
            return self._fallback_summary(text, max_length)

    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Synchronous wrapper for generate_summary_async"""
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.generate_summary_async(text, max_length)
            )
        except Exception as e:
            logger.error(f"Error in synchronous summary generation: {e}")
            return self._fallback_summary(text, max_length)

    def _fallback_summary(self, text: str, max_length: int = 200) -> str:
        """Fallback summary generation using simple extractive method"""
        sentences = text.split(". ")
        summary = ". ".join(sentences[:3])

        if len(summary) > max_length:
            summary = summary[: max_length - 3] + "..."

        return summary

    async def categorize_document_async(self, filename: str, text: str) -> str:
        """Automatically categorize document using Qwen3"""
        if not self.ollama_client:
            return self._fallback_categorization(filename, text)

        try:
            category = await self.ollama_client.categorize_document(filename, text)
            return category
        except Exception as e:
            logger.error(f"Error categorizing document with Ollama: {e}")
            return self._fallback_categorization(filename, text)

    def categorize_document(self, filename: str, text: str) -> str:
        """Synchronous wrapper for categorize_document_async"""
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.categorize_document_async(filename, text)
            )
        except Exception as e:
            logger.error(f"Error in synchronous categorization: {e}")
            return self._fallback_categorization(filename, text)

    def _fallback_categorization(self, filename: str, text: str) -> str:
        """Fallback categorization using keyword matching"""
        filename_lower = filename.lower()
        text_lower = text.lower()

        # Simple keyword-based categorization
        if any(
            word in filename_lower or word in text_lower
            for word in ["faculty", "staff", "professor", "teacher"]
        ):
            return "faculty"
        elif any(
            word in filename_lower or word in text_lower
            for word in ["student", "admission", "grade", "course"]
        ):
            return "student"
        elif any(
            word in filename_lower or word in text_lower
            for word in ["hostel", "accommodation", "room", "residence"]
        ):
            return "hostel"
        elif any(
            word in filename_lower or word in text_lower
            for word in ["academic", "curriculum", "syllabus", "exam"]
        ):
            return "academics"
        elif any(
            word in filename_lower or word in text_lower
            for word in ["mess", "food", "dining", "meal"]
        ):
            return "mess"
        else:
            return "academics"  # Default category
