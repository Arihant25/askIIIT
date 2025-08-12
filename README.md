# Jagruti System Design

## Architecture

The system has three core models:

- **Chatbot** – Interacts with the user, interprets queries, and provides answers.
- **Embedding Model** – Converts documents into vector embeddings for semantic search.
- **Summariser** – Generates structured metadata and concise descriptions for new documents.

We use **ChromaDB** to store all document embeddings along with their metadata.  
ChromaDB enables fast similarity search and supports filtering by metadata (e.g., category, date, author).

---

## Workflow

### 1. Document Creation & Storage

1. A new document is uploaded to the intranet.
2. **Summariser** generates structured metadata:
   - Document name
   - Date of creation
   - Date of last modification
   - Category (e.g., Faculty Data, Student Data, Hostels, Academics, Messes)
   - A brief description of the content
3. **Embedding Model** processes the document:
   - Tokenises and chunks the text into manageable sizes.
   - Generates vector embeddings for each chunk.
4. The embeddings and their metadata are stored in **ChromaDB**.

---

### 2. User Query & Response

1. The user submits a question via the website.
2. **Chatbot**:
   - Converts the query into an embedding.
   - Uses **ChromaDB** to retrieve the most relevant chunks, filtered if needed by metadata.
   - Reads the retrieved content and composes a natural-language answer.
3. The interface displays:
   - Chatbot’s answer.
   - **Highlighted Matches** – exact sentence/phrase from the matched chunk.
   - References to relevant documents.

---

### 3. Website & API Layer

#### Frontend (Next.js)

- Responsive UI for desktop and mobile.
- User authentication and session management.
- Chat interface for queries and responses.
- Metadata-based filtering (category, date range, author, “recently added”).
- **Quick Filters & Facets** for faster narrowing of results.
- **Keyboard Shortcuts** – direct typing focuses chatbox.
- Document preview thumbnails (PDF, text, image icons).
- Clickable links to intranet documents.

#### Backend API (FastAPI)

- Acts as a bridge between frontend, ChromaDB, and AI models.
- Endpoints for:
  - Document upload
  - Metadata retrieval
  - Search queries
  - Embedding creation
  - Summarisation
  - Chatbot queries
- Returns JSON for integration with Next.js.
- Handles filtering and sorting before passing to chatbot.

#### Integration

- Next.js frontend ↔ FastAPI backend via REST.
- FastAPI interacts with ChromaDB and AI models.
- Authentication via **Institute CAS** for secure access.

---

## Admin Panel

Accessible only via CAS authentication.

Features:

- Document upload form with metadata fields.
- **Upload Progress & Status** – from upload → summarising → embedding → ready.
- List of uploaded documents with:
  - Status
  - Last update time
  - Quick re-upload option

---

## Database Schema

ChromaDB is metadata-driven. We maintain two main collections:

### 1. `documents` (metadata only, 1 row per document)

| Field Name  | Type     | Description                                       |
| ----------- | -------- | ------------------------------------------------- |
| doc_id      | UUID     | Unique ID for the document                        |
| name        | String   | Document name                                     |
| category    | Enum     | One of: faculty, student, hostel, academics, mess |
| description | String   | Summary from Summariser                           |
| created_at  | DateTime | Document creation date                            |
| updated_at  | DateTime | Last modification date                            |
| file_url    | String   | Link to intranet file                             |
| author      | String   | Optional uploader’s name                          |

### 2. `chunks` (embeddings + chunk metadata, 1 row per chunk)

| Field Name | Type          | Description                                |
| ---------- | ------------- | ------------------------------------------ |
| chunk_id   | UUID          | Unique ID for the chunk                    |
| doc_id     | UUID          | FK to `documents.doc_id`                   |
| chunk_text | Text          | Actual chunk text                          |
| embedding  | Vector[float] | Vector representation from Embedding Model |
| position   | Int           | Chunk order in document                    |
| category   | Enum          | Same as document category                  |

---

## ChromaDB Notes

- `embedding` is stored in the vector index for similarity search.
- Metadata fields like `category` and `doc_id` are indexed for filtering.

---

## Models

We are using:

- **Qwen 3 (8B)** via Ollama for summarising and chatbot tasks
- **Qwen/Qwen3-Embedding-8B** via sentence-transformers for embeddings
- **pdfplumber** for PDF text extraction (replacing PyPDF2)

## Technical Implementation

### PDF Processing

- Uses **pdfplumber** for robust PDF text extraction
- Handles complex PDF layouts better than PyPDF2
- Preserves text formatting and structure

### Embedding Generation

- **sentence-transformers** with Qwen-based embedding model
- Generates high-quality vector representations for semantic search
- Embeddings are normalized for better similarity calculations

### Chat & Summarization

- **Ollama** with Qwen3:8b model for natural language tasks
- Handles document summarization during upload
- Provides contextual responses using retrieved document chunks

## Configuration

Copy `.env.example` to `.env` and configure:

- Ollama server URL and model settings
- Embedding model preferences
- ChromaDB storage location
- File upload limits and allowed types

## Assumptions

- All data must be machine-readable for LLMs.
- Images require alt-text or descriptive metadata.
- Handwritten text must be OCR-processed before upload.
- Only supported formats (PDF, plain text, etc.) are allowed.
