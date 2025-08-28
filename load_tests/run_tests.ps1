#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Load testing runner script for askIIIT application

.DESCRIPTION
    This script provides easy commands to run various types of load tests
    on the askIIIT application. It can run Locust tests, performance tests,
    stress tests, and streaming tests.

.PARAMETER TestType
    Type of test to run: locust, performance, stress, streaming, all

.PARAMETER Duration
    Duration for load tests in seconds (default: 300)

.PARAMETER Users
    Number of concurrent users for Locust tests (default: 10)

.PARAMETER SpawnRate
    Rate of spawning users per second (default: 2)

.PARAMETER Host
    Target host URL (default: http://localhost:8000)

.EXAMPLE
    .\run_tests.ps1 -TestType locust -Users 20 -Duration 600
    .\run_tests.ps1 -TestType performance
    .\run_tests.ps1 -TestType all
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("locust", "performance", "stress", "streaming", "all")]
    [string]$TestType = "locust",
    
    [Parameter(Mandatory=$false)]
    [int]$Duration = 300,
    
    [Parameter(Mandatory=$false)]
    [int]$Users = 10,
    
    [Parameter(Mandatory=$false)]
    [int]$SpawnRate = 2,
    
    [Parameter(Mandatory=$false)]
    [string]$Host = "http://localhost:8000",
    
    [Parameter(Mandatory=$false)]
    [switch]$WebUI,
    
    [Parameter(Mandatory=$false)]
    [switch]$Install
)

# Color output functions
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$ForegroundColor = "White"
    )
    Write-Host $Message -ForegroundColor $ForegroundColor
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput $Message "Green"
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput $Message "Yellow"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput $Message "Red"
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput $Message "Cyan"
}

# Check if Python is installed
function Test-PythonInstallation {
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Python found: $pythonVersion"
            return $true
        }
    } catch {
        Write-Error "Python not found. Please install Python 3.8 or higher."
        return $false
    }
    return $false
}

# Install dependencies
function Install-Dependencies {
    Write-Info "Installing load testing dependencies..."
    
    if (Test-Path "requirements.txt") {
        python -m pip install -r requirements.txt
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Dependencies installed successfully"
        } else {
            Write-Error "Failed to install dependencies"
            exit 1
        }
    } else {
        Write-Error "requirements.txt not found"
        exit 1
    }
}

# Check if backend is running
function Test-BackendConnection {
    param([string]$Url)
    
    Write-Info "Checking backend connection to $Url..."
    
    try {
        $response = Invoke-WebRequest -Uri "$Url/health" -Method GET -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Success "Backend is responding"
            return $true
        }
    } catch {
        Write-Warning "Backend not responding at $Url"
        Write-Warning "Make sure the askIIIT backend is running"
        return $false
    }
    return $false
}

# Run Locust tests
function Start-LocustTests {
    param(
        [string]$HostUrl,
        [int]$UserCount,
        [int]$SpawnRateValue,
        [int]$TestDuration,
        [bool]$ShowWebUI
    )
    
    Write-Info "Starting Locust load tests..."
    Write-Info "Host: $HostUrl"
    Write-Info "Users: $UserCount"
    Write-Info "Spawn Rate: $SpawnRateValue/s"
    Write-Info "Duration: $TestDuration seconds"
    
    $locustArgs = @(
        "-f", "locustfile.py",
        "--host", $HostUrl,
        "--users", $UserCount,
        "--spawn-rate", $SpawnRateValue,
        "--run-time", "${TestDuration}s",
        "--html", "locust_report.html",
        "--csv", "locust_results"
    )
    
    if (-not $ShowWebUI) {
        $locustArgs += "--headless"
    }
    
    try {
        & locust @locustArgs
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Locust tests completed successfully"
            Write-Info "Report generated: locust_report.html"
        } else {
            Write-Error "Locust tests failed"
        }
    } catch {
        Write-Error "Failed to run Locust tests: $_"
    }
}

# Run performance tests
function Start-PerformanceTests {
    Write-Info "Starting performance tests with pytest..."
    
    try {
        python -m pytest performance_tests.py -v --tb=short
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Performance tests completed successfully"
        } else {
            Write-Error "Performance tests failed"
        }
    } catch {
        Write-Error "Failed to run performance tests: $_"
    }
}

# Run stress tests
function Start-StressTests {
    Write-Info "Starting stress tests..."
    
    try {
        python stress_tests.py
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Stress tests completed successfully"
            Write-Info "Report generated: stress_test_report.json"
        } else {
            Write-Error "Stress tests failed"
        }
    } catch {
        Write-Error "Failed to run stress tests: $_"
    }
}

# Run streaming tests
function Start-StreamingTests {
    Write-Info "Starting streaming performance tests..."
    
    try {
        python streaming_tests.py
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Streaming tests completed successfully"
            Write-Info "Report generated: streaming_performance_report.json"
        } else {
            Write-Error "Streaming tests failed"
        }
    } catch {
        Write-Error "Failed to run streaming tests: $_"
    }
}

# Generate combined report
function New-CombinedReport {
    Write-Info "Generating combined test report..."
    
    $reportScript = @"
import json
import time
from pathlib import Path

def combine_reports():
    reports = {}
    timestamp = time.time()
    
    # Load individual reports
    report_files = [
        ('locust', 'locust_results_stats.csv'),
        ('stress', 'stress_test_report.json'),
        ('streaming', 'streaming_performance_report.json'),
        ('performance', 'performance_report.json')
    ]
    
    for report_type, filename in report_files:
        if Path(filename).exists():
            if filename.endswith('.json'):
                with open(filename, 'r') as f:
                    reports[report_type] = json.load(f)
            else:
                # Handle CSV files (basic)
                reports[report_type] = f"See {filename}"
    
    # Create combined report
    combined = {
        'timestamp': timestamp,
        'test_summary': {
            'reports_included': list(reports.keys()),
            'total_reports': len(reports)
        },
        'individual_reports': reports
    }
    
    with open('combined_test_report.json', 'w') as f:
        json.dump(combined, f, indent=2)
    
    print("Combined report generated: combined_test_report.json")

if __name__ == "__main__":
    combine_reports()
"@
    
    $reportScript | Out-File -FilePath "temp_report_generator.py" -Encoding UTF8
    
    try {
        python temp_report_generator.py
        Remove-Item "temp_report_generator.py" -Force
        Write-Success "Combined report generated: combined_test_report.json"
    } catch {
        Write-Warning "Could not generate combined report: $_"
    }
}

# Main execution logic
Write-Info "askIIIT Load Testing Suite"
Write-Info "=========================="

# Install dependencies if requested
if ($Install) {
    if (-not (Test-PythonInstallation)) {
        exit 1
    }
    Install-Dependencies
    Write-Success "Setup completed. You can now run tests."
    exit 0
}

# Check prerequisites
if (-not (Test-PythonInstallation)) {
    Write-Error "Python installation required"
    exit 1
}

# Check if dependencies are installed
try {
    python -c "import locust" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Dependencies not installed. Run with -Install flag first."
        exit 1
    }
} catch {
    Write-Warning "Dependencies not installed. Run with -Install flag first."
    exit 1
}

# Test backend connection
if (-not (Test-BackendConnection -Url $Host)) {
    $continue = Read-Host "Backend not responding. Continue anyway? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        Write-Info "Please start the askIIIT backend server and try again."
        exit 1
    }
}

# Run selected tests
switch ($TestType) {
    "locust" {
        Start-LocustTests -HostUrl $Host -UserCount $Users -SpawnRateValue $SpawnRate -TestDuration $Duration -ShowWebUI $WebUI
    }
    "performance" {
        Start-PerformanceTests
    }
    "stress" {
        Start-StressTests
    }
    "streaming" {
        Start-StreamingTests
    }
    "all" {
        Write-Info "Running comprehensive test suite..."
        
        # Run all test types
        Start-PerformanceTests
        Start-LocustTests -HostUrl $Host -UserCount $Users -SpawnRateValue $SpawnRate -TestDuration $Duration -ShowWebUI $false
        Start-StressTests
        Start-StreamingTests
        
        # Generate combined report
        New-CombinedReport
        
        Write-Success "All tests completed!"
    }
}

Write-Info "Load testing completed. Check the generated reports for results."
