import numpy as np
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import re
import json
import pdfplumber


from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data if not present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


@dataclass
class Config:
    """Configuration class for embedding generation."""

    pdf_dir: str = "pdfs"
    embeddings_dir: str = "embeddings"
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    chunk_size: int = 500  # target characters per chunk
    chunk_overlap: int = 50
    batch_size: int = 8
    max_chunk_size: int = 1000  # hard limit to prevent overly long chunks


class PDFEmbeddingGenerator:
    """Class to handle PDF text extraction and embedding generation."""

    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        self.load_model()

    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create necessary directories."""
        Path(self.config.embeddings_dir).mkdir(exist_ok=True)

    def load_model(self):
        """Load the sentence transformer model."""
        try:
            self.logger.info(f"Loading model: {self.config.model_name}")
            self.model = SentenceTransformer(self.config.model_name)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using pdfplumber only."""
        text = ""
        if pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception as e:
                self.logger.warning(f"pdfplumber failed for {pdf_path}: {e}")
        else:
            self.logger.error("pdfplumber is not installed. Cannot extract text.")
        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove excessive newlines
        text = re.sub(r"\n+", "\n", text)
        return text.strip()

    def chunk_text_smart(self, text: str) -> List[str]:
        """Smart chunking that respects sentence boundaries."""
        if not text.strip():
            return []

        # Split into sentences
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed chunk size, finalize current chunk
            if (
                len(current_chunk) + len(sentence) > self.config.chunk_size
                and current_chunk
            ):
                chunks.append(current_chunk.strip())

                # Start new chunk with overlap if applicable
                if self.config.chunk_overlap > 0 and chunks:
                    overlap_text = current_chunk[-self.config.chunk_overlap :].strip()
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

            # Handle very long sentences that exceed max chunk size
            if len(current_chunk) > self.config.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk[: self.config.max_chunk_size].strip())
                current_chunk = current_chunk[self.config.max_chunk_size :].strip()

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise

    def process_pdf(self, pdf_path: Path) -> Optional[str]:
        """Process a single PDF file."""
        save_path = Path(self.config.embeddings_dir) / f"{pdf_path.name}.npz"

        if save_path.exists():
            self.logger.info(f"Embeddings for {pdf_path.name} already exist, skipping")
            return "skipped"

        self.logger.info(f"Processing {pdf_path.name}...")

        try:
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                self.logger.warning(f"No text found in {pdf_path.name}")
                return "no_text"

            # Create chunks
            chunks = self.chunk_text_smart(text)
            if not chunks:
                self.logger.warning(f"No chunks created for {pdf_path.name}")
                return "no_chunks"

            self.logger.info(f"  Created {len(chunks)} chunks")

            # Generate embeddings
            embeddings = self.generate_embeddings(chunks)

            # Save results
            np.savez_compressed(
                save_path,
                texts=np.array(chunks, dtype=object),
                embeddings=embeddings,
                metadata=json.dumps(
                    {
                        "source_file": pdf_path.name,
                        "chunk_count": len(chunks),
                        "model_name": self.config.model_name,
                    }
                ),
            )

            self.logger.info(f"  Saved embeddings to {save_path}")
            return "success"

        except Exception as e:
            self.logger.error(f"Error processing {pdf_path.name}: {e}")
            return "error"

    def process_all_pdfs(self):
        """Process all PDF files in the directory."""
        pdf_dir = Path(self.config.pdf_dir)

        if not pdf_dir.exists():
            self.logger.error(f"PDF directory {pdf_dir} does not exist")
            return

        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {pdf_dir}")
            return

        total_files = len(pdf_files)
        results = {"success": 0, "skipped": 0, "error": 0, "no_text": 0, "no_chunks": 0}

        self.logger.info(f"Found {total_files} PDF files to process")

        for idx, pdf_path in enumerate(pdf_files, 1):
            self.logger.info(f"Progress: {idx}/{total_files}")
            result = self.process_pdf(pdf_path)
            if result:
                results[result] += 1

        # Summary
        self.logger.info("Processing complete!")
        self.logger.info(f"Results: {dict(results)}")


def main():
    """Main function to run the PDF embedding generation."""
    config = Config()
    generator = PDFEmbeddingGenerator(config)
    generator.process_all_pdfs()


if __name__ == "__main__":
    main()
