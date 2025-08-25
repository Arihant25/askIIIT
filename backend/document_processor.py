"""
Document processing utilities for extracting text and creating embeddings
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import pdfplumber

# Suppress pdfminer warnings about invalid color patterns
logging.getLogger('pdfminer.pdfinterp').setLevel(logging.ERROR)
import io
from sentence_transformers import SentenceTransformer
import chromadb
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import memory optimization utilities
try:
    from memory_config import MemoryMonitor, setup_memory_optimized_environment
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

# Initialize spaCy for advanced tokenization
def _init_spacy():
    """Initialize spaCy with fallback options"""
    try:
        import spacy
        # Try to load the model, fallback to basic tokenization if not available
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy with en_core_web_sm model for advanced tokenization")
            return nlp
        except OSError:
            # Model not installed, use basic spacy tokenizer
            try:
                nlp = spacy.blank("en")
                logger.warning("Using basic spaCy tokenizer. For better performance, install: python -m spacy download en_core_web_sm")
                return nlp
            except Exception:
                logger.warning("spaCy not available, falling back to basic tokenization")
                return None
    except ImportError:
        logger.warning("spaCy not installed, falling back to basic tokenization")
        return None

# Global spaCy instance
nlp = _init_spacy()


class DocumentProcessor:
    def __init__(self, embedding_model_name: Optional[str] = None, max_workers: Optional[int] = None, memory_config: Optional[Dict] = None, force_cpu: bool = False):
        """Initialize document processor with Qwen3 embedding model and parallel processing optimizations"""
        
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

            # OPTIMIZED: Allow parallel processing while maintaining memory safety
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

            # PARALLEL PROCESSING OPTIMIZATION: Configure based on device and available resources
            if device == "cuda":
                # Conservative GPU settings - multiple workers but smaller batches
                self.max_workers = max_workers or min(4, os.cpu_count() or 2)
                self.embedding_batch_size = 2  # Small batches per worker
                self.model_batch_size = 2
            elif device == "mps":
                # MPS settings - moderate parallelization
                self.max_workers = max_workers or min(3, os.cpu_count() or 2)
                self.embedding_batch_size = 4
                self.model_batch_size = 4
            else:
                # CPU settings - more aggressive parallelization
                self.max_workers = max_workers or min(6, os.cpu_count() or 4)
                self.embedding_batch_size = 8
                self.model_batch_size = 8
                
            self.chunk_processing_batch_size = memory_config.get("chunk_processing_batch_size", 20)
            self.chunk_size = memory_config.get("chunk_size", 400)
            self.chunk_overlap = memory_config.get("chunk_overlap", 50)
            self.device = device

            # Thread-safe model access
            self._model_lock = threading.RLock()
            self._worker_semaphore = threading.Semaphore(self.max_workers)

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
            logger.info(f"Parallel processing enabled: {self.max_workers} workers, batch_size={self.embedding_batch_size}, device={device}")
            
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
        """Split text into overlapping chunks using spaCy tokenization for better semantic boundaries"""
        if not text:
            return []

        # Use memory-optimized chunk sizes
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap

        if nlp:
            # Use spaCy for advanced tokenization
            try:
                # Process text with spaCy to get sentences and tokens
                doc = nlp(text)
                
                # First, try to chunk by sentences for better semantic boundaries
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                
                if sentences:
                    chunks = self._chunk_by_sentences(sentences, chunk_size, overlap)
                    if chunks:
                        logger.debug(f"Created {len(chunks)} chunks from {len(sentences)} sentences using spaCy")
                        return chunks
                
                # Fallback to token-based chunking if sentence chunking doesn't work well
                tokens = [token.text for token in doc if not token.is_space]
                chunks = self._chunk_by_tokens(tokens, chunk_size, overlap)
                logger.debug(f"Created {len(chunks)} chunks from {len(tokens)} tokens using spaCy")
                return chunks
                
            except Exception as e:
                logger.warning(f"Error using spaCy tokenization, falling back to basic split: {e}")
                # Fall through to basic tokenization
        
        # Fallback to basic word splitting if spaCy is not available
        words = text.split()
        chunks = self._chunk_by_tokens(words, chunk_size, overlap)
        logger.debug(f"Created {len(chunks)} chunks from {len(words)} words using basic tokenization")
        return chunks

    def _chunk_by_sentences(self, sentences: List[str], chunk_size: int, overlap: int) -> List[str]:
        """Create chunks based on sentences, respecting word count limits"""
        chunks = []
        current_chunk_words = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If adding this sentence would exceed chunk_size, finalize current chunk
            if current_word_count + sentence_word_count > chunk_size and current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk_words) > overlap:
                    current_chunk_words = current_chunk_words[-overlap:]
                    current_word_count = len(current_chunk_words)
                else:
                    current_chunk_words = []
                    current_word_count = 0
            
            # Add sentence to current chunk
            current_chunk_words.extend(sentence_words)
            current_word_count += sentence_word_count
        
        # Add final chunk if it has content
        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
        
        return chunks

    def _chunk_by_tokens(self, tokens: List[str], chunk_size: int, overlap: int) -> List[str]:
        """Create chunks based on tokens/words with overlap"""
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i: i + chunk_size]
            chunk_text = " ".join(chunk_tokens)
            chunks.append(chunk_text)
            
            # Stop if we've reached the end
            if i + chunk_size >= len(tokens):
                break
        
        return chunks

    def _process_embedding_batch(self, texts_batch: List[str]) -> List[List[float]]:
        """Process a batch of texts for embedding generation with thread-safe access"""
        try:
            with self._worker_semaphore:  # Limit concurrent workers
                with self._model_lock:  # Ensure thread-safe access to the model
                    # OPTIMIZED: Use configured batch size for parallel processing
                    embeddings = self.embedding_model.encode(
                        texts_batch,
                        convert_to_tensor=False,
                        show_progress_bar=False,
                        batch_size=self.embedding_batch_size,
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

    def _process_single_embedding(self, text: str, worker_id: int = 0) -> List[float]:
        """Process a single text for embedding generation with error handling"""
        try:
            with self._worker_semaphore:
                with self._model_lock:
                    # Clear GPU cache before processing
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
                        if hasattr(torch.mps, 'synchronize'):
                            torch.mps.synchronize()
                    
                    # Process single text
                    with torch.no_grad():
                        embedding = self.embedding_model.encode(
                            [text],
                            convert_to_tensor=False,
                            show_progress_bar=False,
                            batch_size=1,
                            normalize_embeddings=True,
                            device=self.device,
                        )
                    
                    if hasattr(embedding, "tolist"):
                        embedding = embedding.tolist()
                    else:
                        embedding = list(embedding)
                    
                    # Clear cache after processing
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    elif self.device == "mps":
                        torch.mps.empty_cache()
                    
                    return embedding[0]
                    
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM in worker {worker_id}, falling back to CPU")
            # Fallback to CPU processing
            return self._process_single_text_cpu(text)
        except Exception as e:
            logger.error(f"Error in worker {worker_id} processing text: {e}")
            raise

    def _process_single_text_cpu(self, text: str) -> List[float]:
        """Fallback CPU processing for a single text"""
        try:
            # Temporarily move model to CPU if needed
            current_device = self.embedding_model.device
            if str(current_device) != "cpu":
                logger.warning("Temporarily switching to CPU for this text")
                temp_model = self.embedding_model.to("cpu")
            else:
                temp_model = self.embedding_model
            
            with torch.no_grad():
                embedding = temp_model.encode(
                    [text],
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=1,
                    normalize_embeddings=True,
                    device="cpu",
                )
            
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            else:
                embedding = list(embedding)
            
            return embedding[0]
            
        except Exception as e:
            logger.error(f"Error in CPU fallback processing: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with parallel processing for improved efficiency"""
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using {self.max_workers} workers (device={self.device})")

            if len(texts) == 0:
                return []

            # For small number of texts or single worker, use sequential processing
            if len(texts) <= self.max_workers or self.max_workers == 1:
                logger.info("Using sequential processing for small batch")
                return self._generate_embeddings_sequential(texts)

            # Use parallel processing for larger batches
            logger.info("Using parallel processing for efficiency")
            return self._generate_embeddings_parallel(texts)

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Fallback to sequential CPU processing
            logger.warning("Falling back to sequential CPU processing")
            return self._process_texts_on_cpu(texts)

    def _generate_embeddings_sequential(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings sequentially (fallback for small batches)"""
        all_embeddings = []
        
        for i, text in enumerate(texts):
            if i % 10 == 0 and i > 0:
                logger.debug(f"Sequential processing: {i}/{len(texts)} completed")
                
            try:
                embedding = self._process_single_embedding(text, worker_id=0)
                all_embeddings.append(embedding)
                
                # Periodic cleanup
                if i % 5 == 0:
                    import gc
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error processing text {i+1}: {e}")
                continue

        return all_embeddings

    def _generate_embeddings_parallel(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using parallel processing with thread pool"""
        all_embeddings = [None] * len(texts)  # Pre-allocate list to maintain order
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="EmbeddingWorker") as executor:
                # Submit all tasks
                future_to_index = {}
                
                for i, text in enumerate(texts):
                    future = executor.submit(self._process_single_embedding, text, i)
                    future_to_index[future] = i
                
                # Process completed tasks
                completed = 0
                failed = 0
                
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    
                    try:
                        embedding = future.result(timeout=30)  # 30 second timeout per text
                        all_embeddings[index] = embedding
                        completed += 1
                        
                        if completed % 10 == 0:
                            logger.debug(f"Parallel processing: {completed}/{len(texts)} completed")
                            
                    except Exception as e:
                        logger.error(f"Failed to process text {index + 1}: {e}")
                        failed += 1
                        
                        # Try CPU fallback for failed items
                        try:
                            cpu_embedding = self._process_single_text_cpu(texts[index])
                            all_embeddings[index] = cpu_embedding
                            logger.info(f"CPU fallback successful for text {index + 1}")
                        except Exception as cpu_e:
                            logger.error(f"CPU fallback also failed for text {index + 1}: {cpu_e}")

                # Filter out None values (failed embeddings)
                valid_embeddings = [emb for emb in all_embeddings if emb is not None]
                
                logger.info(f"Parallel processing completed: {len(valid_embeddings)} successful, {failed} failed")
                return valid_embeddings

        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            # Fallback to sequential processing
            logger.warning("Falling back to sequential processing")
            return self._generate_embeddings_sequential(texts)

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
        """Process texts on CPU with optimized batching for parallel processing"""
        try:
            logger.info(f"Processing {len(texts)} texts on CPU with parallel batching")
            
            # Use larger batches for CPU since it has more memory
            cpu_batch_size = min(16, len(texts))
            all_embeddings = []
            
            # Process in batches using thread pool for CPU
            with ThreadPoolExecutor(max_workers=min(4, self.max_workers), thread_name_prefix="CPUWorker") as executor:
                # Create batches
                batches = []
                for i in range(0, len(texts), cpu_batch_size):
                    batch_texts = texts[i:i + cpu_batch_size]
                    batches.append(batch_texts)
                
                # Submit batch processing tasks
                future_to_batch = {}
                for i, batch in enumerate(batches):
                    future = executor.submit(self._process_cpu_batch, batch, i)
                    future_to_batch[future] = i
                
                # Collect results in order
                batch_results = [None] * len(batches)
                
                for future in as_completed(future_to_batch):
                    batch_index = future_to_batch[future]
                    try:
                        batch_embeddings = future.result(timeout=60)  # 60 second timeout per batch
                        batch_results[batch_index] = batch_embeddings
                        logger.debug(f"CPU batch {batch_index + 1}/{len(batches)} completed")
                    except Exception as e:
                        logger.error(f"CPU batch {batch_index + 1} failed: {e}")
                        batch_results[batch_index] = []
                
                # Flatten results
                for batch_embeddings in batch_results:
                    if batch_embeddings:
                        all_embeddings.extend(batch_embeddings)
            
            logger.info(f"CPU processing completed: {len(all_embeddings)} embeddings generated")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error processing texts on CPU: {e}")
            # Final fallback to sequential CPU processing
            return self._sequential_cpu_fallback(texts)

    def _process_cpu_batch(self, batch_texts: List[str], batch_id: int) -> List[List[float]]:
        """Process a batch of texts on CPU"""
        try:
            with torch.no_grad():
                # Ensure model is on CPU for this thread
                if str(self.embedding_model.device) != "cpu":
                    # Create a temporary CPU copy for this thread
                    temp_model = SentenceTransformer(
                        self.embedding_model.model_name or "Qwen/Qwen3-Embedding-0.6B",
                        device="cpu",
                        trust_remote_code=True
                    )
                else:
                    temp_model = self.embedding_model
                
                embeddings = temp_model.encode(
                    batch_texts,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=8,  # Reasonable CPU batch size
                    normalize_embeddings=True,
                    device="cpu",
                )
                
                if hasattr(embeddings, "tolist"):
                    batch_embeddings = embeddings.tolist()
                else:
                    batch_embeddings = list(embeddings)
                
                # Cleanup
                del embeddings
                import gc
                gc.collect()
                
                return batch_embeddings
                
        except Exception as e:
            logger.error(f"Error in CPU batch {batch_id}: {e}")
            return []

    def _sequential_cpu_fallback(self, texts: List[str]) -> List[List[float]]:
        """Final fallback: sequential CPU processing without threading"""
        try:
            logger.warning("Using sequential CPU fallback (no threading)")
            
            all_embeddings = []
            
            # Ensure model is on CPU
            if str(self.embedding_model.device) != "cpu":
                self.embedding_model = self.embedding_model.to("cpu")
            
            for i, text in enumerate(texts):
                try:
                    with torch.no_grad():
                        embedding = self.embedding_model.encode(
                            [text],
                            convert_to_tensor=False,
                            show_progress_bar=False,
                            batch_size=1,
                            normalize_embeddings=True,
                            device="cpu",
                        )
                    
                    if hasattr(embedding, "tolist"):
                        embedding = embedding.tolist()
                    else:
                        embedding = list(embedding)
                    
                    all_embeddings.append(embedding[0])
                    
                    if i % 20 == 0 and i > 0:
                        logger.debug(f"Sequential CPU fallback: {i}/{len(texts)} completed")
                        import gc
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"Sequential fallback failed for text {i+1}: {e}")
                    continue
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Sequential CPU fallback failed: {e}")
            return []

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
