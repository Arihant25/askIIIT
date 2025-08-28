"""
Performance testing using pytest and aiohttp for async load testing.
Complements Locust tests with more detailed performance analysis.
"""

import pytest
import asyncio
import aiohttp
import time
import statistics
import json
import random
from typing import List, Dict, Any, Tuple
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

from config import config, test_data, MockAuthHeaders

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Stores performance test results"""
    endpoint: str
    response_times: List[float] = field(default_factory=list)
    status_codes: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0
    
    @property
    def median_response_time(self) -> float:
        return statistics.median(self.response_times) if self.response_times else 0
    
    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index]
    
    @property
    def success_rate(self) -> float:
        if not self.status_codes:
            return 0
        successful = sum(1 for code in self.status_codes if 200 <= code < 300)
        return (successful / len(self.status_codes)) * 100
    
    @property
    def error_rate(self) -> float:
        return 100 - self.success_rate


class AsyncPerformanceTester:
    """Async performance testing framework"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.base_url
        self.session = None
        self.metrics: Dict[str, PerformanceMetrics] = {}
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=MockAuthHeaders.get_headers()
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Dict[str, Any] = None,
        params: Dict[str, Any] = None,
        headers: Dict[str, str] = None
    ) -> Tuple[int, float, Dict[str, Any]]:
        """Make a single async request and measure performance"""
        
        if endpoint not in self.metrics:
            self.metrics[endpoint] = PerformanceMetrics(endpoint=endpoint)
        
        url = f"{self.base_url}{endpoint}"
        
        # Record system metrics before request
        process = psutil.Process()
        cpu_before = process.cpu_percent()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        status_code = 0
        response_data = {}
        error_msg = None
        
        try:
            request_headers = headers or MockAuthHeaders.get_headers()
            
            if method.upper() == "GET":
                async with self.session.get(url, params=params, headers=request_headers) as response:
                    status_code = response.status
                    response_data = await response.json()
            elif method.upper() == "POST":
                async with self.session.post(url, json=data, params=params, headers=request_headers) as response:
                    status_code = response.status
                    response_data = await response.json()
            
        except asyncio.TimeoutError:
            error_msg = "Request timeout"
            status_code = 408
        except aiohttp.ClientError as e:
            error_msg = f"Client error: {str(e)}"
            status_code = 500
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            status_code = 500
        
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Record system metrics after request
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Store metrics
        metrics = self.metrics[endpoint]
        metrics.response_times.append(response_time)
        metrics.status_codes.append(status_code)
        metrics.memory_usage.append(memory_after - memory_before)
        metrics.cpu_usage.append(cpu_after - cpu_before)
        
        if error_msg:
            metrics.errors.append(error_msg)
        
        return status_code, response_time, response_data
    
    async def concurrent_test(
        self, 
        method: str, 
        endpoint: str,
        num_requests: int = 50,
        concurrency: int = 10,
        data_generator=None,
        params_generator=None
    ) -> PerformanceMetrics:
        """Run concurrent requests to test performance under load"""
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def make_single_request(request_id: int):
            async with semaphore:
                data = data_generator() if data_generator else None
                params = params_generator() if params_generator else None
                return await self.make_request(method, endpoint, data=data, params=params)
        
        # Run all requests concurrently
        tasks = [make_single_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        logger.info(f"Concurrent test completed: {len(successful_results)} successful, {len(failed_results)} failed")
        
        return self.metrics.get(endpoint, PerformanceMetrics(endpoint=endpoint))


@pytest.fixture
async def perf_tester():
    """Pytest fixture for performance tester"""
    async with AsyncPerformanceTester() as tester:
        yield tester


@pytest.mark.asyncio
class TestEndpointPerformance:
    """Performance tests for individual endpoints"""
    
    async def test_health_endpoint_performance(self, perf_tester):
        """Test health endpoint performance"""
        metrics = await perf_tester.concurrent_test(
            "GET", "/health", 
            num_requests=100, 
            concurrency=20
        )
        
        # Performance assertions
        assert metrics.avg_response_time < 200, f"Health endpoint too slow: {metrics.avg_response_time}ms"
        assert metrics.success_rate > 99, f"Health endpoint success rate too low: {metrics.success_rate}%"
        assert metrics.p95_response_time < 500, f"Health endpoint P95 too high: {metrics.p95_response_time}ms"
        
        logger.info(f"Health endpoint performance: avg={metrics.avg_response_time:.2f}ms, success={metrics.success_rate:.1f}%")
    
    async def test_search_performance(self, perf_tester):
        """Test search endpoint performance"""
        
        def search_data_generator():
            return test_data.generate_search_query()
        
        metrics = await perf_tester.concurrent_test(
            "POST", "/api/search",
            num_requests=50,
            concurrency=10,
            data_generator=search_data_generator
        )
        
        # Performance assertions for search (more lenient as it involves ML processing)
        assert metrics.avg_response_time < 5000, f"Search too slow: {metrics.avg_response_time}ms"
        assert metrics.success_rate > 95, f"Search success rate too low: {metrics.success_rate}%"
        assert metrics.p95_response_time < 10000, f"Search P95 too high: {metrics.p95_response_time}ms"
        
        logger.info(f"Search performance: avg={metrics.avg_response_time:.2f}ms, success={metrics.success_rate:.1f}%")
    
    async def test_chat_performance(self, perf_tester):
        """Test chat endpoint performance"""
        
        def chat_data_generator():
            return test_data.generate_chat_message()
        
        metrics = await perf_tester.concurrent_test(
            "POST", "/api/chat",
            num_requests=30,
            concurrency=5,  # Lower concurrency for chat due to LLM processing
            data_generator=chat_data_generator
        )
        
        # Performance assertions for chat (most lenient due to LLM processing)
        assert metrics.avg_response_time < 15000, f"Chat too slow: {metrics.avg_response_time}ms"
        assert metrics.success_rate > 90, f"Chat success rate too low: {metrics.success_rate}%"
        assert metrics.p95_response_time < 30000, f"Chat P95 too high: {metrics.p95_response_time}ms"
        
        logger.info(f"Chat performance: avg={metrics.avg_response_time:.2f}ms, success={metrics.success_rate:.1f}%")
    
    async def test_document_list_performance(self, perf_tester):
        """Test document listing performance"""
        
        def params_generator():
            return {
                "limit": random.randint(10, 50),
                "offset": random.randint(0, 20)
            }
        
        metrics = await perf_tester.concurrent_test(
            "GET", "/api/documents",
            num_requests=60,
            concurrency=15,
            params_generator=params_generator
        )
        
        # Performance assertions
        assert metrics.avg_response_time < 1000, f"Document list too slow: {metrics.avg_response_time}ms"
        assert metrics.success_rate > 99, f"Document list success rate too low: {metrics.success_rate}%"
        assert metrics.p95_response_time < 2000, f"Document list P95 too high: {metrics.p95_response_time}ms"
        
        logger.info(f"Document list performance: avg={metrics.avg_response_time:.2f}ms, success={metrics.success_rate:.1f}%")


@pytest.mark.asyncio
class TestScenarioPerformance:
    """Performance tests for realistic user scenarios"""
    
    async def test_typical_user_session(self, perf_tester):
        """Test a typical user session flow"""
        session_start = time.time()
        
        # 1. Health check
        await perf_tester.make_request("GET", "/health")
        
        # 2. Get categories
        await perf_tester.make_request("GET", "/api/categories")
        
        # 3. List documents
        await perf_tester.make_request("GET", "/api/documents", params={"limit": 20})
        
        # 4. Perform searches
        for _ in range(3):
            search_data = test_data.generate_search_query()
            await perf_tester.make_request("POST", "/api/search", data=search_data)
            await asyncio.sleep(0.5)  # Simulate user thinking time
        
        # 5. Chat conversation
        conversation_id = None
        conversation_history = []
        
        for i in range(5):
            chat_data = test_data.generate_chat_message(conversation_id)
            if conversation_history:
                chat_data["conversation_history"] = conversation_history[-10:]
            
            status, _, response = await perf_tester.make_request("POST", "/api/chat", data=chat_data)
            
            if status == 200 and "conversation_id" in response:
                conversation_id = response["conversation_id"]
                conversation_history.append({
                    "type": "user",
                    "content": chat_data["message"]
                })
                conversation_history.append({
                    "type": "bot",
                    "content": response.get("response", "")
                })
            
            await asyncio.sleep(1)  # Simulate user reading time
        
        session_duration = time.time() - session_start
        
        # Session should complete in reasonable time
        assert session_duration < 60, f"User session too long: {session_duration:.2f}s"
        
        logger.info(f"Typical user session completed in {session_duration:.2f}s")
    
    async def test_concurrent_users_scenario(self, perf_tester):
        """Test multiple concurrent user sessions"""
        
        async def simulate_user_session(user_id: int):
            """Simulate a single user session"""
            try:
                # Shorter user session for concurrent testing
                await perf_tester.make_request("GET", "/health")
                
                # Search
                search_data = test_data.generate_search_query()
                await perf_tester.make_request("POST", "/api/search", data=search_data)
                
                # Chat
                chat_data = test_data.generate_chat_message()
                await perf_tester.make_request("POST", "/api/chat", data=chat_data)
                
                # List documents
                await perf_tester.make_request("GET", "/api/documents", params={"limit": 10})
                
                return f"User {user_id} completed successfully"
                
            except Exception as e:
                return f"User {user_id} failed: {str(e)}"
        
        # Simulate 20 concurrent users
        num_users = 20
        tasks = [simulate_user_session(i) for i in range(num_users)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        # Check results
        successful_users = sum(1 for r in results if isinstance(r, str) and "completed successfully" in r)
        success_rate = (successful_users / num_users) * 100
        
        assert success_rate > 80, f"Concurrent users success rate too low: {success_rate:.1f}%"
        assert duration < 30, f"Concurrent users test took too long: {duration:.2f}s"
        
        logger.info(f"Concurrent users test: {successful_users}/{num_users} successful in {duration:.2f}s")


@pytest.mark.asyncio
class TestMemoryAndResource:
    """Tests for memory usage and resource consumption"""
    
    async def test_memory_leak_detection(self, perf_tester):
        """Test for potential memory leaks during sustained operation"""
        
        # Force garbage collection before test
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Perform many operations
        for batch in range(10):
            # Batch of search operations
            for _ in range(10):
                search_data = test_data.generate_search_query()
                await perf_tester.make_request("POST", "/api/search", data=search_data)
            
            # Batch of chat operations
            for _ in range(5):
                chat_data = test_data.generate_chat_message()
                await perf_tester.make_request("POST", "/api/chat", data=chat_data)
            
            # Check memory usage every batch
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory
            
            # Log memory usage
            logger.info(f"Batch {batch}: Memory usage {current_memory:.2f}MB (+{memory_increase:.2f}MB)")
            
            # Force garbage collection
            gc.collect()
        
        # Final memory check
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert total_increase < 500, f"Potential memory leak: {total_increase:.2f}MB increase"
        
        logger.info(f"Memory leak test completed: {total_increase:.2f}MB total increase")
    
    async def test_resource_cleanup(self, perf_tester):
        """Test that resources are properly cleaned up"""
        
        # Monitor open file descriptors and connections
        process = psutil.Process()
        initial_connections = len(process.connections())
        initial_open_files = process.num_fds() if hasattr(process, 'num_fds') else 0
        
        # Create many short-lived requests
        for _ in range(50):
            await perf_tester.make_request("GET", "/health")
            await perf_tester.make_request("GET", "/api/categories")
        
        # Wait a bit for cleanup
        await asyncio.sleep(2)
        
        # Check resource usage
        final_connections = len(process.connections())
        final_open_files = process.num_fds() if hasattr(process, 'num_fds') else 0
        
        connection_increase = final_connections - initial_connections
        file_increase = final_open_files - initial_open_files
        
        # Resources should not accumulate excessively
        assert connection_increase < 10, f"Too many open connections: {connection_increase}"
        assert file_increase < 20, f"Too many open files: {file_increase}"
        
        logger.info(f"Resource cleanup test: {connection_increase} new connections, {file_increase} new files")


def generate_performance_report(metrics: Dict[str, PerformanceMetrics], output_file: str = "performance_report.json"):
    """Generate a comprehensive performance report"""
    
    report = {
        "timestamp": time.time(),
        "summary": {
            "total_endpoints_tested": len(metrics),
            "overall_avg_response_time": statistics.mean([m.avg_response_time for m in metrics.values()]),
            "overall_success_rate": statistics.mean([m.success_rate for m in metrics.values()]),
        },
        "endpoints": {}
    }
    
    for endpoint, metric in metrics.items():
        report["endpoints"][endpoint] = {
            "avg_response_time": metric.avg_response_time,
            "median_response_time": metric.median_response_time,
            "p95_response_time": metric.p95_response_time,
            "success_rate": metric.success_rate,
            "error_rate": metric.error_rate,
            "total_requests": len(metric.response_times),
            "total_errors": len(metric.errors),
            "avg_memory_delta": statistics.mean(metric.memory_usage) if metric.memory_usage else 0,
            "avg_cpu_delta": statistics.mean(metric.cpu_usage) if metric.cpu_usage else 0,
        }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Performance report saved to {output_file}")
    return report


if __name__ == "__main__":
    """Run performance tests directly"""
    async def run_performance_tests():
        async with AsyncPerformanceTester() as tester:
            logger.info("Starting performance tests...")
            
            # Run basic endpoint tests
            await tester.concurrent_test("GET", "/health", 100, 20)
            await tester.concurrent_test("GET", "/api/categories", 50, 10)
            
            # Generate performance report
            report = generate_performance_report(tester.metrics)
            
            # Print summary
            print(f"\nPerformance Test Summary:")
            print(f"Endpoints tested: {report['summary']['total_endpoints_tested']}")
            print(f"Overall avg response time: {report['summary']['overall_avg_response_time']:.2f}ms")
            print(f"Overall success rate: {report['summary']['overall_success_rate']:.1f}%")
    
    asyncio.run(run_performance_tests())
