# askIIIT

A comprehensive toolkit for extracting, embedding, and querying information from IIIT-related documents, including PDFs and web sources. This project is designed to support research, data analysis, and information retrieval for IIIT Hyderabad and related academic, administrative, and residential resources.

## Features

- **Web Scraping**: Scrapes relevant web pages to supplement document-based information.
- **PDF Processing**: Extracts text and generates embeddings from a large collection of IIIT-related PDFs for downstream search and analysis.
- **Chat Interface**: Provides a conversational interface for querying the processed data.
- **Embeddings Generation**: Converts document text into vector embeddings for semantic search and retrieval.

## Directory Structure

```
├── chat.py                  # Chatbot interface for querying processed data
├── pdf_to_embeddings.py     # PDF text extraction and embedding generation
├── scraper.py               # Web scraping utility
├── pdfs/                    # Collection of IIIT-related PDF documents
```

## Usage

1. **Web Scraping**
   - Use `scraper.py` to collect additional data from relevant web sources.
2. **PDF Embedding Generation**
   - Run `pdf_to_embeddings.py` to process all PDFs in the `pdfs/` directory and generate embeddings.
3. **Chatbot Querying**
   - Start `chat.py` to interactively query the processed data using natural language.

## Setup

1. Clone the repository:

   ```pwsh
   git clone <repo-url>
   cd askIIIT
   ```

2. Install dependencies:

   ```pwsh
   pip install -r requirements.txt
   ```

3. Place all relevant PDFs in the `pdfs/` directory.
