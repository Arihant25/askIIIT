# Load Testing for askIIIT Application

This directory contains comprehensive load testing tools for the askIIIT application, including tests for the FastAPI backend and document search/chat functionality.

## Test Types

### 1. **Locust Tests** (`locustfile.py`)
- **Purpose**: Web-based load testing with realistic user behavior simulation
- **Features**:
  - Multiple user classes (regular users, admin users, heavy users)
  - Realistic conversation flows with memory
  - Web UI for real-time monitoring
  - Comprehensive endpoint coverage

### 2. **Performance Tests** (`performance_tests.py`)
- **Purpose**: Async performance testing with detailed metrics
- **Features**:
  - Concurrent request testing
  - Response time analysis (avg, median, P95)
  - Memory and CPU usage monitoring
  - Success rate validation

### 3. **Stress Tests** (`stress_tests.py`)
- **Purpose**: System behavior under extreme load
- **Features**:
  - Sustained high-load testing
  - Spike testing (sudden traffic increases)
  - Endurance testing (long-duration)
  - Resource usage monitoring

### 4. **Streaming Tests** (`streaming_tests.py`)
- **Purpose**: Real-time streaming chat performance
- **Features**:
  - WebSocket-like streaming simulation
  - Chunk delivery timing analysis
  - Concurrent streaming sessions
  - Memory leak detection

## Quick Start

### Prerequisites
- Python 3.8 or higher
- askIIIT backend running on `http://localhost:8000`

### Installation
```powershell
# Install dependencies
.\run_tests.ps1 -Install

# Or manually:
pip install -r requirements.txt
```

### Basic Usage

#### Run Locust Tests (Recommended for beginners)
```powershell
# Basic load test with web UI
.\run_tests.ps1 -TestType locust -WebUI

# Headless load test
.\run_tests.ps1 -TestType locust -Users 20 -Duration 300
```

#### Run Performance Tests
```powershell
.\run_tests.ps1 -TestType performance
```

#### Run All Tests
```powershell
.\run_tests.ps1 -TestType all
```

## Detailed Usage

### Locust Web Interface
When using `-WebUI` flag, Locust starts a web interface at `http://localhost:8089` where you can:
- Start/stop tests dynamically
- Monitor real-time metrics
- View response time charts
- Download detailed reports

### Configuration

#### Environment Variables
```powershell
# Set custom backend URL
$env:LOAD_TEST_BASE_URL = "http://your-server:8000"

# Set custom admin users for testing
$env:LOAD_TEST_ADMIN_USERS = "admin@iiit.ac.in,test@iiit.ac.in"
```

#### Test Parameters in `config.py`
```python
# Modify these values for different test scenarios
config.min_wait = 1000  # Min wait time between requests (ms)
config.max_wait = 3000  # Max wait time between requests (ms)
config.spawn_rate = 2   # Users spawned per second
```

### Advanced Testing Scenarios

#### 1. Normal Load Testing
```powershell
# Simulate 50 concurrent users for 10 minutes
.\run_tests.ps1 -TestType locust -Users 50 -Duration 600
```

#### 2. Stress Testing
```powershell
# Test system limits
.\run_tests.ps1 -TestType stress
```

#### 3. Endurance Testing
```powershell
# Long-running test (modify stress_tests.py)
python stress_tests.py  # Choose option 3 for endurance
```

#### 4. Streaming Performance
```powershell
# Test chat streaming under load
.\run_tests.ps1 -TestType streaming
```

## Test Scenarios

### User Behavior Patterns

#### Regular User (`AskIIITUser`)
- **Weight**: 10 (most common)
- **Behavior**:
  - Document searches (10 tasks)
  - Chat conversations (15 tasks)
  - Streaming chat (8 tasks)
  - Document listing (5 tasks)
  - System info checks (4 tasks)

#### Admin User (`AdminUser`)
- **Weight**: 1 (rare)
- **Behavior**:
  - All regular user actions
  - System administration (3 tasks)
  - Log viewing (2 tasks)
  - User management (1 task)
  - RAG diagnostics (1 task)

#### Heavy User (`HeavyUser`)
- **Weight**: 1 (rare)
- **Behavior**:
  - Rapid consecutive searches (20 tasks)
  - Extended conversations (15 tasks)
  - Very short wait times (0.1-0.5s)

### Endpoints Tested

#### Public Endpoints
- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /api/chat` - Chat with documents
- `POST /api/chat/stream` - Streaming chat
- `POST /api/search` - Document search

#### Authenticated Endpoints
- `GET /api/documents` - List documents
- `GET /api/categories` - Get categories
- `GET /api/stats` - System statistics
- `GET /auth/user` - User information

#### Admin Endpoints
- `GET /api/admin/system-info` - System information
- `GET /api/admin/logs` - Application logs
- `GET /api/admin/users` - User management
- `POST /api/admin/bulk-process` - Bulk processing
- `GET /api/admin/debug/rag-diagnostics` - RAG diagnostics

## Reports and Analysis

### Generated Reports

#### 1. Locust Report (`locust_report.html`)
- **Contains**: Response times, failure rates, request distribution
- **Format**: Interactive HTML with charts
- **Usage**: Open in browser for detailed analysis

#### 2. Performance Report (`performance_report.json`)
- **Contains**: Async performance metrics, memory usage
- **Format**: JSON with structured data
- **Usage**: Programmatic analysis, CI/CD integration

#### 3. Stress Test Report (`stress_test_report.json`)
- **Contains**: System behavior under extreme load
- **Format**: JSON with performance breakdown
- **Usage**: Capacity planning, bottleneck identification

#### 4. Streaming Report (`streaming_performance_report.json`)
- **Contains**: Real-time streaming metrics
- **Format**: JSON with timing analysis
- **Usage**: Streaming performance optimization

#### 5. Combined Report (`combined_test_report.json`)
- **Contains**: All test results in one file
- **Format**: Unified JSON structure
- **Usage**: Comprehensive analysis, reporting

### Key Metrics to Monitor

#### Response Times
- **Target**: < 200ms for health checks
- **Target**: < 5s for search operations
- **Target**: < 15s for chat operations
- **Alert**: P95 > 2x average response time

#### Success Rates
- **Target**: > 99% for health endpoints
- **Target**: > 95% for search endpoints  
- **Target**: > 90% for chat endpoints
- **Alert**: Any endpoint < 90% success rate

#### System Resources
- **Memory**: Monitor for leaks (increase > 500MB over time)
- **CPU**: Should not sustain > 80% during normal load
- **Connections**: Check for proper cleanup

#### Streaming Metrics
- **First Response Time**: < 2s for chat streaming
- **Chunk Intervals**: Consistent timing (< 500ms variance)
- **Connection Success**: > 95% successful streams

## Troubleshooting

### Common Issues

#### 1. Backend Not Responding
```powershell
# Check if backend is running
curl http://localhost:8000/health

# Start backend if needed
cd ../backend
python main.py
```

#### 2. Authentication Errors
- Verify `MockAuthHeaders` configuration in `config.py`
- Check admin user list in environment variables
- Ensure CAS authentication is properly mocked

#### 3. High Failure Rates
- Reduce concurrent users
- Increase wait times between requests
- Check backend logs for errors
- Verify database connections

#### 4. Memory Issues During Testing
```powershell
# Monitor system resources
Get-Process python | Select-Object ProcessName, CPU, WorkingSet
```

#### 5. Streaming Test Failures
- Check network connectivity
- Verify streaming endpoint is working manually
- Reduce concurrent streams for testing

### Performance Tuning

#### For Better Performance
1. **Increase Connection Limits**:
   ```python
   # In performance_tests.py, increase connector limits
   connector = aiohttp.TCPConnector(limit=200, limit_per_host=100)
   ```

2. **Optimize Request Timing**:
   ```python
   # In config.py, reduce wait times
   config.min_wait = 500
   config.max_wait = 1500
   ```

3. **Concurrent User Scaling**:
   ```powershell
   # Gradually increase users to find limits
   .\run_tests.ps1 -Users 10   # Start small
   .\run_tests.ps1 -Users 25   # Increase gradually  
   .\run_tests.ps1 -Users 50   # Find breaking point
   ```

#### For Realistic Testing
1. **Simulate Network Delays**:
   ```python
   # Add artificial delays in test code
   await asyncio.sleep(random.uniform(0.1, 0.3))
   ```

2. **Vary Request Patterns**:
   ```python
   # Different user behaviors at different times
   if time.time() % 3600 < 1800:  # First half hour
       weight = 20  # Higher activity
   else:
       weight = 5   # Lower activity
   ```

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Load Tests
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          cd load_tests
          pip install -r requirements.txt
      
      - name: Start backend
        run: |
          cd backend
          python main.py &
          sleep 10
      
      - name: Run performance tests
        run: |
          cd load_tests
          python -m pytest performance_tests.py --junitxml=results.xml
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: load-test-results
          path: load_tests/*.json
```

### Automated Monitoring
```python
# Add to your monitoring system
def check_performance_thresholds(report_file):
    with open(report_file) as f:
        data = json.load(f)
    
    # Define thresholds
    thresholds = {
        'avg_response_time': 5000,  # 5 seconds
        'success_rate': 95,         # 95%
        'error_rate': 5             # 5%
    }
    
    # Check and alert
    for metric, threshold in thresholds.items():
        if data['summary'][metric] > threshold:
            send_alert(f"Performance threshold exceeded: {metric}")
```

## Best Practices

### 1. Test Planning
- Start with small loads and gradually increase
- Test different time periods (peak/off-peak simulation)
- Include both success and failure scenarios
- Plan for different user types and behaviors

### 2. Environment Setup
- Use dedicated test environments
- Ensure consistent system resources
- Monitor both client and server resources
- Use realistic test data

### 3. Result Analysis
- Focus on percentiles, not just averages
- Look for trends over time, not just snapshots
- Correlate performance with system resources
- Identify bottlenecks systematically

### 4. Continuous Testing
- Integrate with deployment pipelines
- Set up automated performance regression detection
- Regular benchmark testing
- Performance budget enforcement

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the generated reports for error details
3. Check backend logs for server-side issues
4. Modify test parameters based on your system capacity

The load testing suite is designed to be comprehensive yet flexible. Start with basic tests and gradually explore more advanced scenarios as you become familiar with the tools.
