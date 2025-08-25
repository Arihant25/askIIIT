#!/usr/bin/env python3
"""
Test script for parallel embedding generation
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List
import asyncio

# Add backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from document_processor import DocumentProcessor
from colored_logging import setup_logging

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_texts(count: int = 50) -> List[str]:
    """Generate test texts for embedding"""
    
    base_texts = [
        "This is a test document about computer science and artificial intelligence.",
        "The university provides excellent facilities for research and development.",
        "Students can access the library resources and online databases.",
        "Faculty members are engaged in cutting-edge research projects.",
        "The campus has modern laboratories and equipment for experiments.",
        "International collaborations enhance the learning experience.",
        "The hostel provides comfortable accommodation for students.",
        "The mess serves nutritious and delicious meals.",
        "Academic calendar includes various examinations and events.",
        "Career guidance helps students in their professional development."
    ]
    
    # Generate variations to create more test data
    test_texts = []
    for i in range(count):
        base_text = base_texts[i % len(base_texts)]
        variation = f"{base_text} Version {i+1} with additional context about topic {i+1}."
        test_texts.append(variation)
    
    return test_texts

def test_sequential_processing(processor: DocumentProcessor, texts: List[str]) -> float:
    """Test sequential embedding generation"""
    
    logger.info("Testing sequential processing...")
    
    # Force sequential by temporarily setting max_workers to 1
    original_workers = processor.max_workers
    processor.max_workers = 1
    
    start_time = time.time()
    embeddings = processor.generate_embeddings(texts)
    end_time = time.time()
    
    # Restore original workers
    processor.max_workers = original_workers
    
    duration = end_time - start_time
    logger.info(f"Sequential processing: {len(embeddings)} embeddings in {duration:.2f}s")
    
    return duration

def test_parallel_processing(processor: DocumentProcessor, texts: List[str]) -> float:
    """Test parallel embedding generation"""
    
    logger.info("Testing parallel processing...")
    
    start_time = time.time()
    embeddings = processor.generate_embeddings(texts)
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"Parallel processing: {len(embeddings)} embeddings in {duration:.2f}s")
    
    return duration

def test_memory_usage():
    """Test memory usage during parallel processing"""
    
    try:
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
        
        return process
        
    except ImportError:
        logger.warning("psutil not available, skipping memory monitoring")
        return None

def main():
    """Main test function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test parallel embedding generation")
    parser.add_argument("--count", type=int, default=50, help="Number of test texts")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU processing")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers (0=auto)")
    
    args = parser.parse_args()
    
    logger.info("🧪 Starting parallel embedding tests...")
    
    # Monitor memory
    memory_monitor = test_memory_usage()
    
    try:
        # Initialize processor
        logger.info("Initializing document processor...")
        
        processor = DocumentProcessor(
            force_cpu=args.force_cpu,
            max_workers=args.workers if args.workers > 0 else None
        )
        
        logger.info(f"Processor configured: device={processor.device}, workers={processor.max_workers}")
        
        if memory_monitor:
            after_init_memory = memory_monitor.memory_info().rss / (1024 * 1024)
            logger.info(f"Memory after initialization: {after_init_memory:.1f} MB")
        
        # Generate test texts
        test_texts = generate_test_texts(args.count)
        logger.info(f"Generated {len(test_texts)} test texts")
        
        # Test sequential processing
        sequential_time = test_sequential_processing(processor, test_texts)
        
        if memory_monitor:
            after_sequential_memory = memory_monitor.memory_info().rss / (1024 * 1024)
            logger.info(f"Memory after sequential: {after_sequential_memory:.1f} MB")
        
        # Test parallel processing
        parallel_time = test_parallel_processing(processor, test_texts)
        
        if memory_monitor:
            after_parallel_memory = memory_monitor.memory_info().rss / (1024 * 1024)
            logger.info(f"Memory after parallel: {after_parallel_memory:.1f} MB")
        
        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE RESULTS")
        print("="*60)
        print(f"Test texts: {args.count}")
        print(f"Device: {processor.device}")
        print(f"Workers: {processor.max_workers}")
        print(f"Batch size: {processor.embedding_batch_size}")
        print("-"*60)
        print(f"Sequential time: {sequential_time:.2f}s")
        print(f"Parallel time: {parallel_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        print("-"*60)
        
        if speedup > 1.2:
            print("✅ Parallel processing is faster!")
        elif speedup > 0.8:
            print("⚠️  Parallel processing is similar to sequential")
        else:
            print("❌ Parallel processing is slower (overhead issues)")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()
