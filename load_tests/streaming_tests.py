"""
WebSocket load testing for streaming chat functionality.
Tests real-time performance of streaming endpoints.
"""

import asyncio
import websockets
import aiohttp
import json
import time
import statistics
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
import random

from config import config, test_data, MockAuthHeaders

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StreamingMetrics:
    """Metrics for streaming performance"""
    connection_times: List[float] = field(default_factory=list)
    first_response_times: List[float] = field(default_factory=list)
    chunk_intervals: List[float] = field(default_factory=list)
    total_chunks: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    successful_streams: int = 0
    failed_streams: int = 0
    
    @property
    def avg_connection_time(self) -> float:
        return statistics.mean(self.connection_times) if self.connection_times else 0
    
    @property
    def avg_first_response_time(self) -> float:
        return statistics.mean(self.first_response_times) if self.first_response_times else 0
    
    @property
    def avg_chunk_interval(self) -> float:
        return statistics.mean(self.chunk_intervals) if self.chunk_intervals else 0
    
    @property
    def success_rate(self) -> float:
        total = self.successful_streams + self.failed_streams
        return (self.successful_streams / total * 100) if total > 0 else 0

class StreamingLoadTester:
    """Load tester for streaming endpoints"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.base_url
        self.session = None
        self.metrics = StreamingMetrics()
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=60, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_streaming_chat_single(self, chat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single streaming chat request"""
        
        connection_start = time.time()
        first_chunk_time = None
        chunk_times = []
        total_chunks = 0
        last_chunk_time = connection_start
        error_msg = None
        
        try:
            headers = MockAuthHeaders.get_headers()
            
            async with self.session.post(
                f"{self.base_url}/api/chat/stream",
                json=chat_data,
                headers=headers
            ) as response:
                
                connection_time = time.time() - connection_start
                self.metrics.connection_times.append(connection_time * 1000)  # ms
                
                if response.status != 200:
                    error_msg = f"HTTP {response.status}"
                    self.metrics.errors.append(error_msg)
                    self.metrics.failed_streams += 1
                    return {"success": False, "error": error_msg}
                
                async for line in response.content:
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            try:
                                data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                                
                                current_time = time.time()
                                
                                if first_chunk_time is None:
                                    first_chunk_time = current_time
                                    first_response_time = (current_time - connection_start) * 1000  # ms
                                    self.metrics.first_response_times.append(first_response_time)
                                else:
                                    chunk_interval = (current_time - last_chunk_time) * 1000  # ms
                                    chunk_times.append(chunk_interval)
                                
                                last_chunk_time = current_time
                                total_chunks += 1
                                
                                # Check if this is the final chunk
                                if data.get("is_final"):
                                    break
                                    
                            except json.JSONDecodeError:
                                continue
                
                # Store metrics
                if chunk_times:
                    self.metrics.chunk_intervals.extend(chunk_times)
                self.metrics.total_chunks.append(total_chunks)
                self.metrics.successful_streams += 1
                
                return {
                    "success": True,
                    "connection_time": connection_time * 1000,
                    "first_response_time": (first_chunk_time - connection_start) * 1000 if first_chunk_time else 0,
                    "total_chunks": total_chunks,
                    "avg_chunk_interval": statistics.mean(chunk_times) if chunk_times else 0
                }
        
        except asyncio.TimeoutError:
            error_msg = "Timeout"
            self.metrics.errors.append(error_msg)
            self.metrics.failed_streams += 1
            return {"success": False, "error": error_msg}
        
        except Exception as e:
            error_msg = str(e)
            self.metrics.errors.append(error_msg)
            self.metrics.failed_streams += 1
            return {"success": False, "error": error_msg}
    
    async def concurrent_streaming_test(
        self, 
        num_streams: int = 10, 
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Test multiple concurrent streaming connections"""
        
        logger.info(f"Starting concurrent streaming test: {num_streams} streams, {max_concurrent} concurrent")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def single_stream_test(stream_id: int):
            async with semaphore:
                chat_data = test_data.generate_chat_message()
                return await self.test_streaming_chat_single(chat_data)
        
        # Run all streams
        tasks = [single_stream_test(i) for i in range(num_streams)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, dict)]
        
        logger.info(f"Concurrent streaming test completed: {len(valid_results)} results")
        
        return valid_results
    
    async def sustained_streaming_test(self, duration_seconds: int = 300, concurrent_streams: int = 3):
        """Test sustained streaming load over time"""
        
        logger.info(f"Starting sustained streaming test: {duration_seconds}s with {concurrent_streams} concurrent streams")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        stream_results = []
        
        async def continuous_streaming():
            while time.time() < end_time:
                try:
                    chat_data = test_data.generate_chat_message()
                    result = await self.test_streaming_chat_single(chat_data)
                    stream_results.append(result)
                    
                    # Small delay between streams from same "user"
                    await asyncio.sleep(random.uniform(5, 15))
                    
                except Exception as e:
                    logger.error(f"Error in sustained streaming: {e}")
        
        # Run multiple concurrent streaming sessions
        tasks = [continuous_streaming() for _ in range(concurrent_streams)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        actual_duration = time.time() - start_time
        
        logger.info(f"Sustained streaming test completed: {len(stream_results)} streams in {actual_duration:.1f}s")
        
        return stream_results

async def test_streaming_performance():
    """Main streaming performance test function"""
    
    async with StreamingLoadTester() as tester:
        logger.info("Starting streaming performance tests...")
        
        # Test 1: Single stream performance
        logger.info("Test 1: Single stream baseline")
        single_result = await tester.test_streaming_chat_single(test_data.generate_chat_message())
        logger.info(f"Single stream result: {single_result}")
        
        # Test 2: Concurrent streams
        logger.info("Test 2: Concurrent streams")
        concurrent_results = await tester.concurrent_streaming_test(num_streams=20, max_concurrent=5)
        
        # Test 3: High concurrency burst
        logger.info("Test 3: High concurrency burst")
        burst_results = await tester.concurrent_streaming_test(num_streams=50, max_concurrent=15)
        
        # Test 4: Sustained load (shorter for demo)
        logger.info("Test 4: Sustained streaming load")
        sustained_results = await tester.sustained_streaming_test(duration_seconds=120, concurrent_streams=3)
        
        # Generate report
        generate_streaming_report(tester.metrics, [
            single_result,
            *concurrent_results,
            *burst_results,
            *sustained_results
        ])
        
        return tester.metrics

def generate_streaming_report(metrics: StreamingMetrics, all_results: List[Dict[str, Any]]):
    """Generate streaming performance report"""
    
    successful_results = [r for r in all_results if r.get("success")]
    failed_results = [r for r in all_results if not r.get("success")]
    
    report = {
        "timestamp": time.time(),
        "summary": {
            "total_streams": len(all_results),
            "successful_streams": len(successful_results),
            "failed_streams": len(failed_results),
            "success_rate": metrics.success_rate,
            "avg_connection_time": metrics.avg_connection_time,
            "avg_first_response_time": metrics.avg_first_response_time,
            "avg_chunk_interval": metrics.avg_chunk_interval,
            "avg_chunks_per_stream": statistics.mean(metrics.total_chunks) if metrics.total_chunks else 0
        },
        "detailed_metrics": {
            "connection_times": {
                "min": min(metrics.connection_times) if metrics.connection_times else 0,
                "max": max(metrics.connection_times) if metrics.connection_times else 0,
                "p95": sorted(metrics.connection_times)[int(0.95 * len(metrics.connection_times))] if metrics.connection_times else 0
            },
            "first_response_times": {
                "min": min(metrics.first_response_times) if metrics.first_response_times else 0,
                "max": max(metrics.first_response_times) if metrics.first_response_times else 0,
                "p95": sorted(metrics.first_response_times)[int(0.95 * len(metrics.first_response_times))] if metrics.first_response_times else 0
            },
            "chunk_intervals": {
                "min": min(metrics.chunk_intervals) if metrics.chunk_intervals else 0,
                "max": max(metrics.chunk_intervals) if metrics.chunk_intervals else 0,
                "p95": sorted(metrics.chunk_intervals)[int(0.95 * len(metrics.chunk_intervals))] if metrics.chunk_intervals else 0
            }
        },
        "errors": {
            "total": len(metrics.errors),
            "unique_errors": list(set(metrics.errors))
        }
    }
    
    # Save report
    with open("streaming_performance_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("STREAMING PERFORMANCE REPORT")
    print(f"{'='*60}")
    print(f"Total streams tested: {report['summary']['total_streams']}")
    print(f"Success rate: {report['summary']['success_rate']:.1f}%")
    print(f"Avg connection time: {report['summary']['avg_connection_time']:.1f}ms")
    print(f"Avg first response time: {report['summary']['avg_first_response_time']:.1f}ms")
    print(f"Avg chunk interval: {report['summary']['avg_chunk_interval']:.1f}ms")
    print(f"Avg chunks per stream: {report['summary']['avg_chunks_per_stream']:.1f}")
    
    if report['errors']['unique_errors']:
        print(f"\nErrors encountered:")
        for error in report['errors']['unique_errors'][:5]:  # Show top 5 errors
            print(f"  - {error}")
    
    print(f"\nDetailed report saved to: streaming_performance_report.json")
    print(f"{'='*60}")
    
    return report

async def streaming_memory_test():
    """Test memory usage during streaming operations"""
    
    import psutil
    import gc
    
    logger.info("Starting streaming memory test...")
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    async with StreamingLoadTester() as tester:
        
        # Run multiple batches of streaming tests
        for batch in range(10):
            logger.info(f"Memory test batch {batch + 1}/10")
            
            # Run concurrent streams
            await tester.concurrent_streaming_test(num_streams=10, max_concurrent=5)
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            logger.info(f"Batch {batch + 1}: Memory {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
            
            # Force garbage collection
            gc.collect()
            await asyncio.sleep(1)  # Allow cleanup
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory
        
        logger.info(f"Streaming memory test completed")
        logger.info(f"Memory increase: {total_increase:.1f}MB")
        
        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": total_increase,
            "memory_leak_detected": total_increase > 200  # Threshold
        }

if __name__ == "__main__":
    """Run streaming tests directly"""
    
    async def main():
        print("Starting askIIIT streaming performance tests...")
        
        test_type = input("Choose test (1=performance, 2=memory): ").strip()
        
        if test_type == "2":
            result = await streaming_memory_test()
            print(f"Memory test result: {result}")
        else:
            metrics = await test_streaming_performance()
            print(f"Streaming performance test completed")
    
    asyncio.run(main())
