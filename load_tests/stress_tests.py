"""
Stress testing scenarios for askIIIT application.
Tests system behavior under extreme load conditions.
"""

import asyncio
import aiohttp
import time
import random
import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
import psutil
import statistics

from config import config, test_data, MockAuthHeaders

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StressTestResult:
    """Results from stress testing"""
    test_name: str
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    requests_per_second: float
    error_rate: float
    peak_memory_mb: float
    peak_cpu_percent: float
    errors: List[str]

class StressTester:
    """Stress testing framework for askIIIT"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or config.base_url
        self.session = None
        self.results: List[StressTestResult] = []
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=200,  # Higher limit for stress testing
            limit_per_host=100,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=60, connect=15)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def stress_test_endpoint(
        self,
        test_name: str,
        method: str,
        endpoint: str,
        duration_seconds: int,
        max_concurrent: int,
        data_generator=None,
        params_generator=None,
        headers_generator=None
    ) -> StressTestResult:
        """Run stress test on a specific endpoint"""
        
        logger.info(f"Starting stress test: {test_name}")
        logger.info(f"Duration: {duration_seconds}s, Max concurrent: {max_concurrent}")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        response_times = []
        status_codes = []
        errors = []
        request_count = 0
        successful_requests = 0
        failed_requests = 0
        
        # System monitoring
        process = psutil.Process()
        peak_memory = 0
        peak_cpu = 0
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def make_request():
            nonlocal request_count, successful_requests, failed_requests, peak_memory, peak_cpu
            
            async with semaphore:
                request_start = time.time()
                
                try:
                    # Generate request data
                    data = data_generator() if data_generator else None
                    params = params_generator() if params_generator else None
                    headers = headers_generator() if headers_generator else MockAuthHeaders.get_headers()
                    
                    url = f"{self.base_url}{endpoint}"
                    
                    # Monitor system resources
                    try:
                        current_memory = process.memory_info().rss / 1024 / 1024  # MB
                        current_cpu = process.cpu_percent()
                        peak_memory = max(peak_memory, current_memory)
                        peak_cpu = max(peak_cpu, current_cpu)
                    except:
                        pass
                    
                    # Make request
                    if method.upper() == "GET":
                        async with self.session.get(url, params=params, headers=headers) as response:
                            status_code = response.status
                            await response.read()  # Consume response
                    elif method.upper() == "POST":
                        async with self.session.post(url, json=data, params=params, headers=headers) as response:
                            status_code = response.status
                            await response.read()  # Consume response
                    else:
                        raise ValueError(f"Unsupported method: {method}")
                    
                    request_time = (time.time() - request_start) * 1000  # ms
                    
                    response_times.append(request_time)
                    status_codes.append(status_code)
                    request_count += 1
                    
                    if 200 <= status_code < 300:
                        successful_requests += 1
                    else:
                        failed_requests += 1
                        errors.append(f"Status {status_code}")
                
                except asyncio.TimeoutError:
                    failed_requests += 1
                    errors.append("Timeout")
                    request_count += 1
                except Exception as e:
                    failed_requests += 1
                    errors.append(str(e))
                    request_count += 1
        
        # Launch requests continuously until duration expires
        tasks = set()
        
        while time.time() < end_time:
            # Launch new request if under concurrency limit
            if len(tasks) < max_concurrent:
                task = asyncio.create_task(make_request())
                tasks.add(task)
            
            # Remove completed tasks
            done_tasks = {task for task in tasks if task.done()}
            tasks -= done_tasks
            
            # Small delay to prevent busy loop
            await asyncio.sleep(0.001)
        
        # Wait for remaining tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        actual_duration = time.time() - start_time
        
        # Calculate metrics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        requests_per_second = request_count / actual_duration if actual_duration > 0 else 0
        error_rate = (failed_requests / request_count * 100) if request_count > 0 else 0
        
        result = StressTestResult(
            test_name=test_name,
            duration=actual_duration,
            total_requests=request_count,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            errors=list(set(errors))  # Unique errors
        )
        
        self.results.append(result)
        
        logger.info(f"Stress test completed: {test_name}")
        logger.info(f"Requests: {request_count}, Success rate: {100-error_rate:.1f}%, RPS: {requests_per_second:.1f}")
        
        return result

async def run_comprehensive_stress_tests():
    """Run a comprehensive suite of stress tests"""
    
    async with StressTester() as tester:
        logger.info("Starting comprehensive stress tests...")
        
        # Test 1: Health endpoint under extreme load
        await tester.stress_test_endpoint(
            test_name="Health Endpoint Extreme Load",
            method="GET",
            endpoint="/health",
            duration_seconds=60,
            max_concurrent=100
        )
        
        # Test 2: Search endpoint stress test
        def search_data_generator():
            return test_data.generate_search_query()
        
        await tester.stress_test_endpoint(
            test_name="Search Endpoint Stress",
            method="POST",
            endpoint="/api/search",
            duration_seconds=120,
            max_concurrent=20,  # Lower for search due to processing
            data_generator=search_data_generator
        )
        
        # Test 3: Chat endpoint sustained load
        def chat_data_generator():
            return test_data.generate_chat_message()
        
        await tester.stress_test_endpoint(
            test_name="Chat Endpoint Sustained Load",
            method="POST",
            endpoint="/api/chat",
            duration_seconds=180,
            max_concurrent=10,  # Even lower for chat
            data_generator=chat_data_generator
        )
        
        # Test 4: Document listing under high concurrency
        def params_generator():
            return {
                "limit": random.randint(10, 100),
                "offset": random.randint(0, 50),
                "categories": random.choice(test_data.categories) if random.random() < 0.5 else None
            }
        
        await tester.stress_test_endpoint(
            test_name="Document List High Concurrency",
            method="GET",
            endpoint="/api/documents",
            duration_seconds=90,
            max_concurrent=50,
            params_generator=params_generator
        )
        
        # Test 5: Mixed workload simulation
        await mixed_workload_stress_test(tester)
        
        # Generate report
        generate_stress_test_report(tester.results)
        
        return tester.results

async def mixed_workload_stress_test(tester: StressTester):
    """Simulate mixed workload with different endpoints"""
    
    logger.info("Starting mixed workload stress test...")
    
    start_time = time.time()
    duration = 300  # 5 minutes
    end_time = start_time + duration
    
    tasks = []
    max_concurrent = 50
    semaphore = asyncio.Semaphore(max_concurrent)
    
    request_counts = {
        "health": 0,
        "search": 0,
        "chat": 0,
        "documents": 0,
        "categories": 0
    }
    
    async def make_mixed_request():
        async with semaphore:
            # Randomly choose endpoint based on realistic usage patterns
            endpoint_choice = random.choices(
                ["health", "search", "chat", "documents", "categories"],
                weights=[5, 25, 30, 25, 15],  # Chat and search are most common
                k=1
            )[0]
            
            request_counts[endpoint_choice] += 1
            
            try:
                if endpoint_choice == "health":
                    async with tester.session.get(f"{tester.base_url}/health") as response:
                        await response.read()
                
                elif endpoint_choice == "search":
                    data = test_data.generate_search_query()
                    headers = MockAuthHeaders.get_headers()
                    async with tester.session.post(f"{tester.base_url}/api/search", json=data, headers=headers) as response:
                        await response.read()
                
                elif endpoint_choice == "chat":
                    data = test_data.generate_chat_message()
                    headers = MockAuthHeaders.get_headers()
                    async with tester.session.post(f"{tester.base_url}/api/chat", json=data, headers=headers) as response:
                        await response.read()
                
                elif endpoint_choice == "documents":
                    params = {"limit": random.randint(10, 50)}
                    headers = MockAuthHeaders.get_headers()
                    async with tester.session.get(f"{tester.base_url}/api/documents", params=params, headers=headers) as response:
                        await response.read()
                
                elif endpoint_choice == "categories":
                    headers = MockAuthHeaders.get_headers()
                    async with tester.session.get(f"{tester.base_url}/api/categories", headers=headers) as response:
                        await response.read()
            
            except Exception as e:
                logger.debug(f"Mixed workload request failed: {e}")
    
    # Launch requests continuously
    while time.time() < end_time:
        if len(tasks) < max_concurrent:
            task = asyncio.create_task(make_mixed_request())
            tasks.append(task)
        
        # Clean up completed tasks
        tasks = [task for task in tasks if not task.done()]
        
        await asyncio.sleep(0.01)  # Small delay
    
    # Wait for remaining tasks
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    
    actual_duration = time.time() - start_time
    total_requests = sum(request_counts.values())
    
    logger.info(f"Mixed workload completed in {actual_duration:.1f}s")
    logger.info(f"Total requests: {total_requests} ({total_requests/actual_duration:.1f} RPS)")
    logger.info(f"Request distribution: {request_counts}")

async def spike_test():
    """Test system behavior under sudden traffic spikes"""
    
    logger.info("Starting spike test...")
    
    async with StressTester() as tester:
        
        # Normal load baseline
        logger.info("Phase 1: Normal load baseline (30s)")
        await tester.stress_test_endpoint(
            test_name="Spike Test - Baseline",
            method="GET",
            endpoint="/health",
            duration_seconds=30,
            max_concurrent=5
        )
        
        # Sudden spike
        logger.info("Phase 2: Traffic spike (60s)")
        await tester.stress_test_endpoint(
            test_name="Spike Test - Spike",
            method="GET",
            endpoint="/health",
            duration_seconds=60,
            max_concurrent=200  # 40x increase
        )
        
        # Return to normal
        logger.info("Phase 3: Return to normal (30s)")
        await tester.stress_test_endpoint(
            test_name="Spike Test - Recovery",
            method="GET",
            endpoint="/health",
            duration_seconds=30,
            max_concurrent=5
        )
        
        return tester.results

async def endurance_test():
    """Long-running test to check for memory leaks and degradation"""
    
    logger.info("Starting endurance test (30 minutes)...")
    
    async with StressTester() as tester:
        
        # Sustained moderate load for 30 minutes
        await tester.stress_test_endpoint(
            test_name="Endurance Test",
            method="GET",
            endpoint="/health",
            duration_seconds=1800,  # 30 minutes
            max_concurrent=10
        )
        
        return tester.results

def generate_stress_test_report(results: List[StressTestResult], output_file: str = "stress_test_report.json"):
    """Generate comprehensive stress test report"""
    
    report = {
        "timestamp": time.time(),
        "summary": {
            "total_tests": len(results),
            "total_requests": sum(r.total_requests for r in results),
            "total_successful": sum(r.successful_requests for r in results),
            "total_failed": sum(r.failed_requests for r in results),
            "avg_rps": statistics.mean([r.requests_per_second for r in results]),
            "avg_error_rate": statistics.mean([r.error_rate for r in results]),
            "peak_memory_mb": max([r.peak_memory_mb for r in results]),
            "peak_cpu_percent": max([r.peak_cpu_percent for r in results])
        },
        "tests": []
    }
    
    for result in results:
        test_report = {
            "name": result.test_name,
            "duration": result.duration,
            "total_requests": result.total_requests,
            "successful_requests": result.successful_requests,
            "failed_requests": result.failed_requests,
            "avg_response_time": result.avg_response_time,
            "max_response_time": result.max_response_time,
            "min_response_time": result.min_response_time,
            "requests_per_second": result.requests_per_second,
            "error_rate": result.error_rate,
            "peak_memory_mb": result.peak_memory_mb,
            "peak_cpu_percent": result.peak_cpu_percent,
            "unique_errors": len(set(result.errors)),
            "error_types": list(set(result.errors))[:10]  # Top 10 error types
        }
        report["tests"].append(test_report)
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("STRESS TEST REPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {report['summary']['total_tests']}")
    print(f"Total requests: {report['summary']['total_requests']:,}")
    print(f"Success rate: {100 - report['summary']['avg_error_rate']:.1f}%")
    print(f"Average RPS: {report['summary']['avg_rps']:.1f}")
    print(f"Peak memory: {report['summary']['peak_memory_mb']:.1f} MB")
    print(f"Peak CPU: {report['summary']['peak_cpu_percent']:.1f}%")
    print(f"\nDetailed report saved to: {output_file}")
    print(f"{'='*60}")
    
    return report

if __name__ == "__main__":
    """Run stress tests directly"""
    
    async def main():
        print("Starting askIIIT stress testing suite...")
        
        # Choose test type
        test_type = input("Choose test type (1=comprehensive, 2=spike, 3=endurance): ").strip()
        
        if test_type == "1":
            results = await run_comprehensive_stress_tests()
        elif test_type == "2":
            results = await spike_test()
        elif test_type == "3":
            results = await endurance_test()
        else:
            print("Running comprehensive tests by default...")
            results = await run_comprehensive_stress_tests()
        
        print(f"\nStress testing completed. {len(results)} tests run.")
    
    asyncio.run(main())
