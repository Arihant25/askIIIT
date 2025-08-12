"""
Document processing utilities for extracting text and creating embeddings
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime
import uuid
import pdfplumber

# Suppress pdfminer warnings about invalid color patterns
logging.getLogger('pdfminer.pdfinterp').setLevel(logging.ERROR)
import io
from sentence_transformers import SentenceTransformer
import chromadb
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial

# Import memory optimization utilities
try:
    from memory_config import MemoryMonitor, MemoryOptimizedConfig, setup_memory_optimized_environment
except ImportError:
    # Fallback if memory_config is not available
    class MemoryMonitor:
        def log_memory_usage(self, context=""): pass
        def check_memory_available(self, required_gb=2.0): return True
        def cleanup_memory(self): 
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def setup_memory_optimized_environment():
        return {
            "embedding_batch_size": 4,
            "chunk_processing_batch_size": 50,
            "max_workers": 2,
            "chunk_size": 500,
            "chunk_overlap": 75,
        }, MemoryMonitor()

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, embedding_model_name: Optional[str] = None, max_workers: Optional[int] = None, memory_config: Optional[Dict] = None, force_cpu: bool = False):
        """Initialize document processor with Qwen3 embedding model and aggressive memory optimizations"""
        
        # Setup memory optimization
        if memory_config is None:
            memory_config, self.memory_monitor = setup_memory_optimized_environment()
        else:
            self.memory_monitor = MemoryMonitor()
        
        self.memory_config = memory_config
        self.force_cpu = force_cpu
        
        try:
            model_name = embedding_model_name or os.getenv(
                "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
            )

            # AGGRESSIVE MEMORY OPTIMIZATION: Force CPU if requested or if GPU memory is too low
            device = "cpu"  # Default to CPU for safety
            
            if not force_cpu:
                if torch.cuda.is_available():
                    # Check available GPU memory
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    available_memory = torch.cuda.memory_reserved(0) / (1024**3)
                    
                    if gpu_memory_gb >= 8:  # Only use GPU if we have enough memory
                        device = "cuda"
                        logger.info(f"Using CUDA for acceleration ({gpu_memory_gb:.1f}GB GPU)")
                    else:
                        logger.warning(f"GPU memory too low ({gpu_memory_gb:.1f}GB), falling back to CPU")
                        device = "cpu"
                elif torch.backends.mps.is_available():
                    device = "mps"
                    logger.info("Using Metal Performance Shaders (MPS) for acceleration")
            
            if device == "cpu":
                logger.info("Using CPU (forced or insufficient GPU memory)")

            trust_remote_code = (
                os.getenv("EMBEDDING_TRUST_REMOTE_CODE",
                          "true").lower() == "true"
            )

            # AGGRESSIVE MEMORY OPTIMIZATION: Ultra-conservative settings for 6GB GPU
            self.max_workers = 1  # Single worker to avoid memory conflicts
            self._lock = threading.Lock()
            
            # Ultra-small batch sizes for GPU memory conservation
            if device == "cuda":
                self.embedding_batch_size = 1  # Process one text at a time on GPU
                self.model_batch_size = 1  # Internal model batch size
            else:
                self.embedding_batch_size = 4  # CPU can handle slightly larger batches
                self.model_batch_size = 4
                
            self.chunk_processing_batch_size = 10  # Very small chunk batches
            self.chunk_size = memory_config.get("chunk_size", 400)  # Smaller chunks
            self.chunk_overlap = memory_config.get("chunk_overlap", 50)  # Less overlap
            self.device = device

            # Set PyTorch memory optimization environment variables
            if device == "cuda":
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
                # Clear any existing GPU memory
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            logger.info(f"Loading embedding model: {model_name} on device: {device}")
            
            # Load model with memory optimization
            self.embedding_model = SentenceTransformer(
                model_name, 
                device=device, 
                trust_remote_code=trust_remote_code,
                cache_folder=os.getenv("TRANSFORMERS_CACHE", None)
            )
            
            # Move model to device explicitly and optimize
            self.embedding_model = self.embedding_model.to(device)
            if device == "cuda":
                self.embedding_model.half()  # Use half precision to save memory
                
            logger.info(f"Successfully loaded Qwen-based embedding model: {model_name}")
            logger.info(f"Ultra-memory optimized settings: batch_size={self.embedding_batch_size}, device={device}, workers=1")
            
            # Log initial memory usage
            if self.memory_monitor:
                self.memory_monitor.log_memory_usage("after model loading")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _extract_page_text(self, page) -> str:
        """Extract text from a single PDF page"""
        try:
            return page.extract_text() or ""
        except Exception as e:
            logger.error(f"Error extracting text from page: {e}")
            return ""

    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF file content using pdfplumber with memory optimization"""
        try:
            pdf_file = io.BytesIO(file_content)
            extracted_text = []

            with pdfplumber.open(pdf_file) as pdf:
                # MEMORY OPTIMIZATION: Process pages sequentially to avoid loading all pages in memory
                total_pages = len(pdf.pages)
                logger.info(f"Processing PDF with {total_pages} pages")
                
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            extracted_text.append(page_text)
                        
                        # MEMORY OPTIMIZATION: Clear page from memory periodically
                        if i % 10 == 0 and i > 0:
                            import gc
                            gc.collect()
                            logger.debug(f"Processed {i}/{total_pages} pages")
                            
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {i}: {e}")
                        continue

                # Join all text and cleanup
                text = "\n".join(extracted_text)
                extracted_text.clear()  # Free memory
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
        self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> List[str]:
        """Split text into overlapping chunks with memory-optimized sizes"""
        if not text:
            return []

        # Use memory-optimized chunk sizes
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i: i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)

            # Stop if we've reached the end
            if i + chunk_size >= len(words):
                break

        logger.debug(f"Created {len(chunks)} chunks from {len(words)} words")
        return chunks

    def _process_embedding_batch(self, texts_batch: List[str]) -> List[List[float]]:
        """Process a batch of texts for embedding generation with memory optimization"""
        try:
            with self._lock:  # Ensure thread-safe access to the model
                # MEMORY OPTIMIZATION: Use smaller batch size and clear GPU cache
                embeddings = self.embedding_model.encode(
                    texts_batch,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=self.embedding_batch_size,  # Much smaller batch size
                    normalize_embeddings=True,
                )

                # Convert to list if numpy array
                if hasattr(embeddings, "tolist"):
                    result = embeddings.tolist()
                else:
                    result = list(embeddings)
                
                # MEMORY OPTIMIZATION: Clear GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    
                return result
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with ultra-aggressive memory optimization for low GPU memory"""
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts (ultra-memory optimized, device={self.device})")

            # ULTRA AGGRESSIVE: Process one text at a time for GPU
            all_embeddings = []
            
            for i, text in enumerate(texts):
                if i % 10 == 0:
                    logger.debug(f"Processing text {i+1}/{len(texts)}")
                    
                try:
                    # Clear GPU cache before each embedding
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
                        if hasattr(torch.mps, 'synchronize'):
                            torch.mps.synchronize()
                    
                    # Process single text with minimal batch size
                    with torch.no_grad():  # Ensure no gradients
                        embedding = self.embedding_model.encode(
                            [text],  # Single text in list
                            convert_to_tensor=False,
                            show_progress_bar=False,
                            batch_size=1,  # Force batch size of 1
                            normalize_embeddings=True,
                            device=self.device,
                        )
                    
                    if hasattr(embedding, "tolist"):
                        embedding = embedding.tolist()
                    else:
                        embedding = list(embedding)
                    
                    # Add the single embedding
                    all_embeddings.append(embedding[0])
                    
                    # AGGRESSIVE: Clear cache after each text
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
                    
                    # Force garbage collection every 5 texts
                    if i % 5 == 0:
                        import gc
                        del embedding
                        gc.collect()
                        
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"GPU OOM on text {i+1}, switching to CPU for remaining texts")
                    # Switch to CPU for remaining texts
                    self._switch_to_cpu()
                    
                    # Process remaining texts on CPU
                    remaining_texts = texts[i:]
                    cpu_embeddings = self._process_texts_on_cpu(remaining_texts)
                    all_embeddings.extend(cpu_embeddings)
                    break
                    
                except Exception as e:
                    logger.error(f"Error processing text {i+1}: {e}")
                    # Skip this text and continue
                    continue

            logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Fallback to CPU processing
            logger.warning("Falling back to CPU processing due to GPU memory issues")
            return self._process_texts_on_cpu(texts)

    def _switch_to_cpu(self):
        """Switch the model to CPU when GPU runs out of memory"""
        try:
            logger.info("Switching embedding model to CPU due to GPU memory constraints")
            self.embedding_model = self.embedding_model.to("cpu")
            self.device = "cpu"
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error switching to CPU: {e}")

    def _process_texts_on_cpu(self, texts: List[str]) -> List[List[float]]:
        """Process texts on CPU with larger batches since CPU has more memory"""
        try:
            logger.info(f"Processing {len(texts)} texts on CPU")
            
            all_embeddings = []
            cpu_batch_size = 8  # Larger batches for CPU
            
            for i in range(0, len(texts), cpu_batch_size):
                batch_texts = texts[i:i + cpu_batch_size]
                
                with torch.no_grad():
                    embeddings = self.embedding_model.encode(
                        batch_texts,
                        convert_to_tensor=False,
                        show_progress_bar=False,
                        batch_size=cpu_batch_size,
                        normalize_embeddings=True,
                        device="cpu",
                    )
                
                if hasattr(embeddings, "tolist"):
                    batch_embeddings = embeddings.tolist()
                else:
                    batch_embeddings = list(embeddings)
                
                all_embeddings.extend(batch_embeddings)
                
                # Cleanup after each batch
                del embeddings, batch_embeddings
                import gc
                gc.collect()
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error processing texts on CPU: {e}")
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
        """Process a complete document with memory optimization: extract text, create chunks, generate embeddings, store in ChromaDB"""
        try:
            logger.info(f"Processing document {filename} (memory optimized)")
            
            # Extract text
            text = self.extract_text_from_file(file_content, filename)

            if not text.strip():
                raise ValueError("No text content found in document")

            # MEMORY OPTIMIZATION: Clear file content from memory immediately
            del file_content
            import gc
            gc.collect()

            # Create chunks
            chunks = self.chunk_text(text)

            if not chunks:
                raise ValueError("No chunks created from document")

            logger.info(f"Created {len(chunks)} chunks for {filename}")

            # MEMORY OPTIMIZATION: Process chunks in smaller batches to avoid memory buildup
            chunk_batch_size = self.chunk_processing_batch_size
            total_chunks_processed = 0
            
            for batch_start in range(0, len(chunks), chunk_batch_size):
                batch_end = min(batch_start + chunk_batch_size, len(chunks))
                chunk_batch = chunks[batch_start:batch_end]
                
                logger.info(f"Processing chunk batch {batch_start//chunk_batch_size + 1}/{(len(chunks) + chunk_batch_size - 1)//chunk_batch_size} for {filename}")
                
                # Generate embeddings for this batch
                batch_embeddings = self.generate_embeddings(chunk_batch)

                if not batch_embeddings:
                    logger.warning(f"No embeddings generated for batch {batch_start//chunk_batch_size + 1}")
                    continue

                # Prepare chunk data for ChromaDB
                chunk_ids = []
                chunk_texts = []
                chunk_embeddings = []
                chunk_metadatas = []

                for i, (chunk_text, embedding) in enumerate(zip(chunk_batch, batch_embeddings)):
                    chunk_id = str(uuid.uuid4())
                    chunk_ids.append(chunk_id)
                    chunk_texts.append(chunk_text)
                    chunk_embeddings.append(embedding)

                    chunk_metadata = {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "position": batch_start + i,
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
                
                total_chunks_processed += len(chunk_batch)
                
                # MEMORY OPTIMIZATION: Clear batch data and force garbage collection
                del chunk_batch, batch_embeddings, chunk_ids, chunk_texts, chunk_embeddings, chunk_metadatas
                gc.collect()
                
                # Clear GPU cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            # Store document metadata (no embedding needed for documents collection)
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

            # Create a dummy embedding for the document (ChromaDB requires it)
            dummy_embedding = [0.0] * 384  # Standard embedding dimension

            documents_collection.add(
                ids=[doc_id],
                documents=[text[:1000]],  # Store first 1000 chars as summary
                embeddings=[dummy_embedding],  # Provide dummy embedding
                metadatas=[doc_metadata],
            )

            # MEMORY OPTIMIZATION: Clear remaining variables
            del text, chunks
            gc.collect()

            logger.info(
                f"Successfully processed document {filename}: {total_chunks_processed} chunks created"
            )

            return {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_count": total_chunks_processed,
                "text_length": doc_metadata["text_length"],
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            # MEMORY OPTIMIZATION: Cleanup on error
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            raise


class DocumentSummarizer:
    """Generate summaries and metadata for documents using Qwen3 via Ollama"""

    def __init__(self):
        try:
            from ollama_client import ollama_client

            self.ollama_client = ollama_client
            logger.info("Initialized document summarizer with Ollama client")
        except Exception as e:
            logger.error(
                f"Failed to initialize Ollama client for summarization: {e}")
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
        """Generate summary with proper async context handling"""
        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            try:
                # Try to get the current event loop
                asyncio.get_running_loop()
                # If there's a running loop, we need to run the async function in a thread
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.generate_summary_async(text, max_length)
                    )
                    return future.result()
            except RuntimeError:
                # No running event loop, safe to create a new one
                return asyncio.run(self.generate_summary_async(text, max_length))
        except Exception as e:
            logger.error(f"Error in summary generation: {e}")
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
        """Categorize document with proper async context handling"""
        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor

            try:
                # Try to get the current event loop
                asyncio.get_running_loop()
                # If there's a running loop, we need to run the async function in a thread
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.categorize_document_async(filename, text)
                    )
                    return future.result()
            except RuntimeError:
                # No running event loop, safe to create a new one
                return asyncio.run(self.categorize_document_async(filename, text))
        except Exception as e:
            logger.error(f"Error in categorization: {e}")
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
