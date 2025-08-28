#!/bin/bash

# Load testing runner script for askIIIT application
#
# This script provides easy commands to run various types of load tests
# on the askIIIT application. It can run Locust tests, performance tests,
# stress tests, and streaming tests.
#
# Usage:
#   ./run_tests.sh [OPTIONS]
#
# Examples:
#   ./run_tests.sh --test-type locust --users 20 --duration 600
#   ./run_tests.sh --test-type performance
#   ./run_tests.sh --test-type all
#   ./run_tests.sh --install

set -e  # Exit on any error

# Default values
TEST_TYPE="locust"
DURATION=300
USERS=10
SPAWN_RATE=2
HOST="http://localhost:8000"
WEB_UI=false
INSTALL=false

# Color output functions
print_color() {
    local color=$1
    local message=$2
    case $color in
        "red")    echo -e "\033[31m$message\033[0m" ;;
        "green")  echo -e "\033[32m$message\033[0m" ;;
        "yellow") echo -e "\033[33m$message\033[0m" ;;
        "cyan")   echo -e "\033[36m$message\033[0m" ;;
        *)        echo "$message" ;;
    esac
}

print_success() { print_color "green" "$1"; }
print_error() { print_color "red" "$1"; }
print_warning() { print_color "yellow" "$1"; }
print_info() { print_color "cyan" "$1"; }

# Help function
show_help() {
    cat << EOF
Load Testing Runner for askIIIT Application

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --test-type TYPE     Type of test to run: locust, performance, stress, streaming, all (default: locust)
    --duration SECONDS   Duration for load tests in seconds (default: 300)
    --users COUNT        Number of concurrent users for Locust tests (default: 10)
    --spawn-rate RATE    Rate of spawning users per second (default: 2)
    --host URL           Target host URL (default: http://localhost:8000)
    --web-ui             Start Locust with web UI (default: headless)
    --install            Install dependencies and exit
    --help               Show this help message

EXAMPLES:
    # Install dependencies
    $0 --install

    # Run Locust tests with web UI
    $0 --test-type locust --users 20 --duration 600 --web-ui

    # Run performance tests
    $0 --test-type performance

    # Run all tests
    $0 --test-type all

    # Run stress tests
    $0 --test-type stress

    # Run streaming tests  
    $0 --test-type streaming

PREREQUISITES:
    - Python 3.8 or higher
    - askIIIT backend running on specified host (default: http://localhost:8000)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-type)
            TEST_TYPE="$2"
            if [[ ! "$TEST_TYPE" =~ ^(locust|performance|stress|streaming|all)$ ]]; then
                print_error "Invalid test type: $TEST_TYPE"
                print_error "Valid types: locust, performance, stress, streaming, all"
                exit 1
            fi
            shift 2
            ;;
        --duration)
            DURATION="$2"
            if ! [[ "$DURATION" =~ ^[0-9]+$ ]]; then
                print_error "Duration must be a positive integer"
                exit 1
            fi
            shift 2
            ;;
        --users)
            USERS="$2"
            if ! [[ "$USERS" =~ ^[0-9]+$ ]]; then
                print_error "Users must be a positive integer"
                exit 1
            fi
            shift 2
            ;;
        --spawn-rate)
            SPAWN_RATE="$2"
            if ! [[ "$SPAWN_RATE" =~ ^[0-9]+$ ]]; then
                print_error "Spawn rate must be a positive integer"
                exit 1
            fi
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --web-ui)
            WEB_UI=true
            shift
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            print_error "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        local python_version=$(python3 --version 2>&1)
        print_success "Python found: $python_version"
        return 0
    elif command -v python &> /dev/null; then
        local python_version=$(python --version 2>&1)
        print_success "Python found: $python_version"
        return 0
    else
        print_error "Python not found. Please install Python 3.8 or higher."
        return 1
    fi
}

# Get python command
get_python_cmd() {
    if command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        echo "python"
    else
        print_error "Python not found"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_info "Installing load testing dependencies..."
    
    local python_cmd=$(get_python_cmd)
    
    if [[ -f "requirements.txt" ]]; then
        $python_cmd -m pip install -r requirements.txt
        if [[ $? -eq 0 ]]; then
            print_success "Dependencies installed successfully"
        else
            print_error "Failed to install dependencies"
            exit 1
        fi
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Check if backend is running
check_backend() {
    local url=$1
    print_info "Checking backend connection to $url..."
    
    if command -v curl &> /dev/null; then
        if curl -s -f "$url/health" > /dev/null 2>&1; then
            print_success "Backend is responding"
            return 0
        fi
    elif command -v wget &> /dev/null; then
        if wget -q --spider "$url/health" 2> /dev/null; then
            print_success "Backend is responding"
            return 0
        fi
    else
        print_warning "Neither curl nor wget found. Cannot check backend connection."
        return 1
    fi
    
    print_warning "Backend not responding at $url"
    print_warning "Make sure the askIIIT backend is running"
    return 1
}

# Run Locust tests
run_locust_tests() {
    local host_url=$1
    local user_count=$2
    local spawn_rate_value=$3
    local test_duration=$4
    local show_web_ui=$5
    
    print_info "Starting Locust load tests..."
    print_info "Host: $host_url"
    print_info "Users: $user_count"
    print_info "Spawn Rate: ${spawn_rate_value}/s"
    print_info "Duration: $test_duration seconds"
    
    local locust_args=(
        "-f" "locustfile.py"
        "--host" "$host_url"
        "--users" "$user_count"
        "--spawn-rate" "$spawn_rate_value"
        "--run-time" "${test_duration}s"
        "--html" "locust_report.html"
        "--csv" "locust_results"
    )
    
    if [[ "$show_web_ui" == "false" ]]; then
        locust_args+=("--headless")
    fi
    
    if locust "${locust_args[@]}"; then
        print_success "Locust tests completed successfully"
        print_info "Report generated: locust_report.html"
    else
        print_error "Locust tests failed"
        return 1
    fi
}

# Run performance tests
run_performance_tests() {
    print_info "Starting performance tests with pytest..."
    
    local python_cmd=$(get_python_cmd)
    
    if $python_cmd -m pytest performance_tests.py -v --tb=short; then
        print_success "Performance tests completed successfully"
    else
        print_error "Performance tests failed"
        return 1
    fi
}

# Run stress tests
run_stress_tests() {
    print_info "Starting stress tests..."
    
    local python_cmd=$(get_python_cmd)
    
    if $python_cmd stress_tests.py; then
        print_success "Stress tests completed successfully"
        print_info "Report generated: stress_test_report.json"
    else
        print_error "Stress tests failed"
        return 1
    fi
}

# Run streaming tests
run_streaming_tests() {
    print_info "Starting streaming performance tests..."
    
    local python_cmd=$(get_python_cmd)
    
    if $python_cmd streaming_tests.py; then
        print_success "Streaming tests completed successfully"
        print_info "Report generated: streaming_performance_report.json"
    else
        print_error "Streaming tests failed"
        return 1
    fi
}

# Generate combined report
generate_combined_report() {
    print_info "Generating combined test report..."
    
    local python_cmd=$(get_python_cmd)
    
    $python_cmd << 'EOF'
import json
import time
from pathlib import Path
import csv

def combine_reports():
    reports = {}
    timestamp = time.time()
    
    # Load individual reports
    report_files = [
        ('stress', 'stress_test_report.json'),
        ('streaming', 'streaming_performance_report.json'),
        ('performance', 'performance_report.json')
    ]
    
    for test_type, filename in report_files:
        if Path(filename).exists():
            try:
                with open(filename, 'r') as f:
                    reports[test_type] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
    
    # Load Locust CSV results if available
    locust_stats_file = 'locust_results_stats.csv'
    if Path(locust_stats_file).exists():
        try:
            locust_data = []
            with open(locust_stats_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    locust_data.append(row)
            reports['locust'] = locust_data
        except Exception as e:
            print(f"Warning: Could not load {locust_stats_file}: {e}")
    
    # Create combined report
    combined_report = {
        'timestamp': timestamp,
        'test_summary': {
            'total_tests': len(reports),
            'test_types': list(reports.keys())
        },
        'results': reports
    }
    
    # Save combined report
    with open('combined_test_report.json', 'w') as f:
        json.dump(combined_report, f, indent=2)
    
    print("Combined report generated: combined_test_report.json")

combine_reports()
EOF
}

# Main execution

# Handle installation
if [[ "$INSTALL" == "true" ]]; then
    if ! check_python; then
        print_error "Python installation required"
        exit 1
    fi
    install_dependencies
    print_success "Setup completed. You can now run tests."
    exit 0
fi

# Check prerequisites
if ! check_python; then
    print_error "Python installation required"
    exit 1
fi

# Check if dependencies are installed
python_cmd=$(get_python_cmd)
if ! $python_cmd -c "import locust" 2>/dev/null; then
    print_warning "Dependencies not installed. Run with --install flag first."
    print_info "Example: $0 --install"
    exit 1
fi

# Test backend connection
if ! check_backend "$HOST"; then
    read -p "Backend not responding. Continue anyway? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Please start the askIIIT backend server and try again."
        print_info "Example: cd ../backend && python main.py"
        exit 1
    fi
fi

# Run selected tests
case "$TEST_TYPE" in
    "locust")
        run_locust_tests "$HOST" "$USERS" "$SPAWN_RATE" "$DURATION" "$WEB_UI"
        ;;
    "performance")
        run_performance_tests
        ;;
    "stress")
        run_stress_tests
        ;;
    "streaming")
        run_streaming_tests
        ;;
    "all")
        print_info "Running comprehensive test suite..."
        
        # Run all test types
        run_performance_tests
        run_locust_tests "$HOST" "$USERS" "$SPAWN_RATE" "$DURATION" "false"
        run_stress_tests
        run_streaming_tests
        
        # Generate combined report
        generate_combined_report
        
        print_success "All tests completed!"
        ;;
esac

print_info "Load testing completed. Check the generated reports for results."
