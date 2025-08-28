"""
CI/CD integration utilities for load testing.
Provides functions for automated testing and reporting.
"""

import os
import json
import time
import subprocess
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadTestCICD:
    """CI/CD integration for load testing"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.results = {}
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load CI/CD configuration"""
        default_config = {
            "thresholds": {
                "avg_response_time": 5000,  # ms
                "success_rate": 95,         # %
                "error_rate": 5,           # %
                "memory_increase": 500,     # MB
                "cpu_usage": 80            # %
            },
            "notification": {
                "enabled": False,
                "email": {
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_email": "loadtest@askiiit.com",
                    "to_emails": []
                }
            },
            "artifacts": {
                "retention_days": 30,
                "upload_to_s3": False,
                "s3_bucket": ""
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def run_performance_test_suite(self) -> Dict[str, Any]:
        """Run complete performance test suite for CI/CD"""
        
        logger.info("Starting CI/CD performance test suite...")
        
        start_time = time.time()
        
        # Test results container
        suite_results = {
            "timestamp": start_time,
            "status": "running",
            "tests": {},
            "summary": {},
            "passed": True,
            "failures": []
        }
        
        try:
            # 1. Quick health check
            logger.info("Running health check...")
            health_result = self._run_health_check()
            suite_results["tests"]["health"] = health_result
            
            if not health_result["passed"]:
                suite_results["passed"] = False
                suite_results["failures"].append("Health check failed")
                return suite_results
            
            # 2. Performance tests
            logger.info("Running performance tests...")
            perf_result = self._run_performance_tests()
            suite_results["tests"]["performance"] = perf_result
            
            if not perf_result["passed"]:
                suite_results["passed"] = False
                suite_results["failures"].extend(perf_result.get("failures", []))
            
            # 3. Light load test
            logger.info("Running light load test...")
            load_result = self._run_light_load_test()
            suite_results["tests"]["load"] = load_result
            
            if not load_result["passed"]:
                suite_results["passed"] = False
                suite_results["failures"].extend(load_result.get("failures", []))
            
            # 4. Generate summary
            suite_results["summary"] = self._generate_test_summary(suite_results["tests"])
            suite_results["duration"] = time.time() - start_time
            suite_results["status"] = "passed" if suite_results["passed"] else "failed"
            
            # 5. Send notifications if configured
            if self.config["notification"]["enabled"]:
                self._send_notification(suite_results)
            
            # 6. Save artifacts
            self._save_artifacts(suite_results)
            
        except Exception as e:
            logger.error(f"Test suite failed with exception: {e}")
            suite_results["status"] = "error"
            suite_results["passed"] = False
            suite_results["error"] = str(e)
        
        return suite_results
    
    def _run_health_check(self) -> Dict[str, Any]:
        """Run basic health check"""
        try:
            import requests
            
            base_url = os.getenv("LOAD_TEST_BASE_URL", "http://localhost:8000")
            
            start_time = time.time()
            response = requests.get(f"{base_url}/health", timeout=10)
            response_time = (time.time() - start_time) * 1000  # ms
            
            passed = (
                response.status_code == 200 and 
                response_time < self.config["thresholds"]["avg_response_time"]
            )
            
            return {
                "passed": passed,
                "status_code": response.status_code,
                "response_time": response_time,
                "threshold": self.config["thresholds"]["avg_response_time"]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run pytest performance tests"""
        try:
            # Run pytest with specific markers for CI
            cmd = ["python", "-m", "pytest", "performance_tests.py", "-v", "--tb=short", "--json-report", "--json-report-file=pytest_report.json"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Parse pytest results
            if Path("pytest_report.json").exists():
                with open("pytest_report.json") as f:
                    pytest_data = json.load(f)
                
                passed = pytest_data["summary"]["passed"] > 0 and pytest_data["summary"]["failed"] == 0
                
                return {
                    "passed": passed,
                    "total_tests": pytest_data["summary"]["total"],
                    "passed_tests": pytest_data["summary"]["passed"],
                    "failed_tests": pytest_data["summary"]["failed"],
                    "duration": pytest_data["duration"],
                    "failures": [test["nodeid"] for test in pytest_data.get("tests", []) if test["outcome"] == "failed"]
                }
            else:
                return {
                    "passed": result.returncode == 0,
                    "output": result.stdout,
                    "error": result.stderr
                }
        
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def _run_light_load_test(self) -> Dict[str, Any]:
        """Run lightweight Locust test for CI"""
        try:
            base_url = os.getenv("LOAD_TEST_BASE_URL", "http://localhost:8000")
            
            # Run short Locust test
            cmd = [
                "locust",
                "-f", "locustfile.py",
                "--host", base_url,
                "--users", "5",
                "--spawn-rate", "1",
                "--run-time", "60s",
                "--headless",
                "--csv", "ci_locust_results"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            # Parse Locust results
            if Path("ci_locust_results_stats.csv").exists():
                import pandas as pd
                
                stats = pd.read_csv("ci_locust_results_stats.csv")
                
                # Calculate metrics
                avg_response_time = stats["Average Response Time"].mean()
                success_rate = (1 - stats["Failure Count"].sum() / stats["Request Count"].sum()) * 100
                
                passed = (
                    avg_response_time < self.config["thresholds"]["avg_response_time"] and
                    success_rate >= self.config["thresholds"]["success_rate"]
                )
                
                failures = []
                if avg_response_time >= self.config["thresholds"]["avg_response_time"]:
                    failures.append(f"Average response time too high: {avg_response_time:.2f}ms")
                if success_rate < self.config["thresholds"]["success_rate"]:
                    failures.append(f"Success rate too low: {success_rate:.1f}%")
                
                return {
                    "passed": passed,
                    "avg_response_time": avg_response_time,
                    "success_rate": success_rate,
                    "total_requests": stats["Request Count"].sum(),
                    "failures": failures
                }
            else:
                return {
                    "passed": result.returncode == 0,
                    "output": result.stdout[:1000],  # Truncate output
                    "error": result.stderr[:1000] if result.stderr else None
                }
        
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    def _generate_test_summary(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all test results"""
        
        total_tests = len(tests)
        passed_tests = sum(1 for test in tests.values() if test.get("passed", False))
        
        return {
            "total_test_suites": total_tests,
            "passed_test_suites": passed_tests,
            "failed_test_suites": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "overall_status": "PASSED" if passed_tests == total_tests else "FAILED"
        }
    
    def _send_notification(self, results: Dict[str, Any]):
        """Send notification about test results"""
        
        if not self.config["notification"]["email"]["to_emails"]:
            logger.warning("No email recipients configured for notifications")
            return
        
        try:
            # Create email content
            subject = f"Load Test Results - {results['status'].upper()}"
            
            body = f"""
Load Test Suite Results

Status: {results['status'].upper()}
Duration: {results.get('duration', 0):.1f} seconds
Timestamp: {time.ctime(results['timestamp'])}

Summary:
- Total Test Suites: {results['summary']['total_test_suites']}
- Passed: {results['summary']['passed_test_suites']}
- Failed: {results['summary']['failed_test_suites']}
- Success Rate: {results['summary']['success_rate']:.1f}%

"""
            
            if results['failures']:
                body += f"\nFailures:\n"
                for failure in results['failures']:
                    body += f"- {failure}\n"
            
            # Send email
            msg = MimeMultipart()
            msg['From'] = self.config["notification"]["email"]["from_email"]
            msg['To'] = ", ".join(self.config["notification"]["email"]["to_emails"])
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            # SMTP configuration
            server = smtplib.SMTP(
                self.config["notification"]["email"]["smtp_server"],
                self.config["notification"]["email"]["smtp_port"]
            )
            
            if self.config["notification"]["email"]["username"]:
                server.starttls()
                server.login(
                    self.config["notification"]["email"]["username"],
                    self.config["notification"]["email"]["password"]
                )
            
            server.send_message(msg)
            server.quit()
            
            logger.info("Notification email sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def _save_artifacts(self, results: Dict[str, Any]):
        """Save test artifacts"""
        
        # Save main results
        with open("ci_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Create artifacts directory
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        # Copy important files to artifacts
        important_files = [
            "ci_test_results.json",
            "pytest_report.json", 
            "ci_locust_results_stats.csv",
            "performance_report.json"
        ]
        
        for file_name in important_files:
            if Path(file_name).exists():
                import shutil
                shutil.copy2(file_name, artifacts_dir / file_name)
        
        logger.info(f"Artifacts saved to {artifacts_dir}")
    
    def validate_thresholds(self, report_file: str) -> Dict[str, Any]:
        """Validate performance against thresholds"""
        
        if not Path(report_file).exists():
            return {"valid": False, "error": "Report file not found"}
        
        try:
            with open(report_file) as f:
                data = json.load(f)
            
            validation_results = {
                "valid": True,
                "violations": [],
                "warnings": []
            }
            
            # Check each threshold
            thresholds = self.config["thresholds"]
            
            if "summary" in data:
                summary = data["summary"]
                
                # Response time check
                if "overall_avg_response_time" in summary:
                    if summary["overall_avg_response_time"] > thresholds["avg_response_time"]:
                        validation_results["violations"].append(
                            f"Average response time ({summary['overall_avg_response_time']:.2f}ms) exceeds threshold ({thresholds['avg_response_time']}ms)"
                        )
                        validation_results["valid"] = False
                
                # Success rate check
                if "overall_success_rate" in summary:
                    if summary["overall_success_rate"] < thresholds["success_rate"]:
                        validation_results["violations"].append(
                            f"Success rate ({summary['overall_success_rate']:.1f}%) below threshold ({thresholds['success_rate']}%)"
                        )
                        validation_results["valid"] = False
            
            return validation_results
            
        except Exception as e:
            return {"valid": False, "error": str(e)}

def create_github_action_summary(results: Dict[str, Any]) -> str:
    """Create GitHub Actions summary markdown"""
    
    status_emoji = "✅" if results["passed"] else "❌"
    
    summary = f"""
## Load Test Results {status_emoji}

**Status**: {results['status'].upper()}  
**Duration**: {results.get('duration', 0):.1f} seconds  
**Timestamp**: {time.ctime(results['timestamp'])}

### Summary
| Metric | Value |
|--------|-------|
| Total Test Suites | {results['summary']['total_test_suites']} |
| Passed | {results['summary']['passed_test_suites']} |
| Failed | {results['summary']['failed_test_suites']} |
| Success Rate | {results['summary']['success_rate']:.1f}% |

"""
    
    if results['failures']:
        summary += "### Failures\n"
        for failure in results['failures']:
            summary += f"- ❌ {failure}\n"
    
    summary += "\n### Test Details\n"
    for test_name, test_result in results['tests'].items():
        test_status = "✅" if test_result.get("passed", False) else "❌"
        summary += f"- {test_status} **{test_name.title()}**: {test_result.get('status', 'Unknown')}\n"
    
    return summary

if __name__ == "__main__":
    """Run CI/CD integration directly"""
    
    # Initialize CI/CD integration
    ci_cd = LoadTestCICD()
    
    # Run test suite
    results = ci_cd.run_performance_test_suite()
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Create GitHub Actions summary if in GitHub Actions environment
    if os.getenv("GITHUB_ACTIONS"):
        summary = create_github_action_summary(results)
        with open(os.getenv("GITHUB_STEP_SUMMARY", "github_summary.md"), "w") as f:
            f.write(summary)
    
    # Exit with appropriate code
    exit(0 if results["passed"] else 1)
