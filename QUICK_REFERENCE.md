# Quick Reference: PDF Embedding Processing

## Quick Start Commands

### 1. System Setup
```bash
# Check system capabilities and get recommendations
python backend/config_parallel.py --info

# Generate optimized settings file
python backend/config_parallel.py --generate

# Validate current configuration
python backend/config_parallel.py --validate
```

### 2. Document Processing
```bash
# Check existing documents in database
python backend/bulk_process.py --check

# Process all PDFs in /pdfs directory
python backend/bulk_process.py --process

# Reset database (CAUTION: Deletes all data)
python backend/bulk_process.py --reset
```

## Configuration Quick Guide

### Environment Variables (.env)
```env
# Essential Settings
CHROMA_PERSIST_DIRECTORY=./chroma_data
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B

# Performance Tuning (adjust based on your system)
FORCE_CPU_EMBEDDINGS=false          # Set to true if GPU issues
EMBEDDING_MAX_WORKERS=4             # Parallel workers (2-8)
EMBEDDING_BATCH_SIZE=4              # Batch size (1-8)
CHUNK_SIZE=400                      # Words per chunk
CHUNK_OVERLAP=50                    # Overlap between chunks
MAX_FILE_SIZE_MB=25                 # Skip larger files
CHUNK_PROCESSING_BATCH_SIZE=20      # Chunks per batch
```

### System-Specific Optimizations

#### High-End GPU System (24GB+ VRAM)
```env
FORCE_CPU_EMBEDDINGS=false
EMBEDDING_MAX_WORKERS=4
EMBEDDING_BATCH_SIZE=2
MAX_FILE_SIZE_MB=50
```

#### Standard GPU System (8GB VRAM)
```env
FORCE_CPU_EMBEDDINGS=false
EMBEDDING_MAX_WORKERS=3
EMBEDDING_BATCH_SIZE=1
MAX_FILE_SIZE_MB=25
```

#### CPU-Only System (16GB+ RAM)
```env
FORCE_CPU_EMBEDDINGS=true
EMBEDDING_MAX_WORKERS=6
EMBEDDING_BATCH_SIZE=8
MAX_FILE_SIZE_MB=25
```

#### Low-Memory System (8GB RAM)
```env
FORCE_CPU_EMBEDDINGS=true
EMBEDDING_MAX_WORKERS=2
EMBEDDING_BATCH_SIZE=4
MAX_FILE_SIZE_MB=10
CHUNK_PROCESSING_BATCH_SIZE=10
```

## Common Issues & Quick Fixes

### Memory Issues
```bash
# Immediate fix - reduce resources
echo "EMBEDDING_MAX_WORKERS=1" >> .env
echo "EMBEDDING_BATCH_SIZE=1" >> .env
echo "CHUNK_PROCESSING_BATCH_SIZE=5" >> .env
```

### GPU Issues
```bash
# Force CPU processing
echo "FORCE_CPU_EMBEDDINGS=true" >> .env
```

### Dimension Mismatch
```bash
# Reset and rebuild database
python backend/bulk_process.py --reset
python backend/bulk_process.py --process
```

### Large File Errors
```bash
# Reduce file size limit
echo "MAX_FILE_SIZE_MB=10" >> .env
```

## File Organization

### Required Directory Structure
```
askIIIT/
├── backend/
│   ├── bulk_process.py
│   ├── config_parallel.py
│   └── .env
├── pdfs/                    # Place PDFs here
│   ├── document1.pdf
│   └── document2.pdf
└── chroma_data/            # Auto-created database
```

### Preparing PDFs
1. **File Size**: Keep under 25MB per file
2. **File Format**: Ensure PDFs contain extractable text
3. **File Names**: Use descriptive names without special characters
4. **Organization**: Group related documents in the pdfs/ directory

## Processing Workflow

### Step-by-Step Process
1. **Place PDFs** in `/pdfs` directory
2. **Check system** with `config_parallel.py --info`
3. **Configure** settings in `.env` file
4. **Validate** with `config_parallel.py --validate`
5. **Process** with `bulk_process.py --process`
6. **Monitor** progress in terminal output

### Processing Output Example
```
🖥️  Processing document.pdf (5.2 MB)
📄 Created 45 chunks from 15 pages
⚡ Generated embeddings using GPU acceleration
✅ Stored in ChromaDB: documents=1, chunks=45
```

## Performance Expectations

| System Type | Speed | Documents/Hour |
|-------------|-------|----------------|
| GPU (24GB) | Fast | 20-30 |
| GPU (8GB) | Medium | 10-20 |
| CPU (16GB) | Slow | 5-15 |
| CPU (8GB) | Very Slow | 3-8 |

## Monitoring Commands

### Check Processing Status
```bash
# View current database contents
python backend/bulk_process.py --check

# Monitor system resources (requires htop)
htop

# Check GPU usage (NVIDIA)
nvidia-smi

# Check disk space
df -h
```

### View Logs
```bash
# Real-time processing logs
tail -f processing.log

# Search for errors
grep -i error *.log

# Check memory warnings
grep -i "memory\|oom" *.log
```

## Emergency Procedures

### System Overload
1. Stop processing: `Ctrl+C`
2. Reduce workers: `EMBEDDING_MAX_WORKERS=1`
3. Enable CPU only: `FORCE_CPU_EMBEDDINGS=true`
4. Restart processing

### Database Corruption
1. Backup existing data: `cp -r chroma_data chroma_data.backup`
2. Reset database: `python backend/bulk_process.py --reset`
3. Reprocess documents: `python backend/bulk_process.py --process`

### Storage Full
1. Check space: `df -h`
2. Clean up: `rm -rf chroma_data/__pycache__`
3. Move old backups: `mv chroma_data.backup /backup/location/`

## Support Resources

### Documentation Files
- `EMBEDDING_PROCESSING_GUIDE.md` - Complete documentation
- `backend/README.md` - Backend-specific information
- `README.md` - Project overview

### Log Files
- Check terminal output for real-time status
- System logs contain detailed error information
- Memory usage tracked during processing

### Configuration Tools
- `config_parallel.py` - System optimization utility
- `bulk_process.py` - Main processing script
- `.env` - Configuration file
