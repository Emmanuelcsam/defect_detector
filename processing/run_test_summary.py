#!/usr/bin/env python3
"""
Run tests and provide a summary of results
"""

import subprocess
import sys
import json
import os

def run_test_module(module_name):
    """Run a specific test module and return results"""
    cmd = ["python", "-m", "unittest", module_name, "-v"]
    
    # Activate virtual environment
    if os.path.exists("test_env"):
        activate_cmd = "source test_env/bin/activate && " + " ".join(cmd)
        result = subprocess.run(activate_cmd, shell=True, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output
    lines = result.stderr.split('\n')
    test_count = 0
    failures = 0
    errors = 0
    
    for line in lines:
        if "Ran" in line and "test" in line:
            parts = line.split()
            test_count = int(parts[1])
        elif "FAILED" in line:
            if "failures=" in line:
                failures = int(line.split("failures=")[1].split(")")[0])
            if "errors=" in line:
                errors = int(line.split("errors=")[1].split(")")[0].split(",")[0])
    
    return {
        "module": module_name,
        "tests": test_count,
        "passed": test_count - failures - errors,
        "failures": failures,
        "errors": errors,
        "success": failures == 0 and errors == 0
    }

def main():
    """Run all test modules and provide summary"""
    print("Running Test Summary")
    print("=" * 60)
    
    test_modules = [
        "tests.test_process_unit",
        "tests.test_separation_unit",
        "tests.test_detection_unit",
        "tests.test_data_acquisition_unit",
        "tests.test_app_unit",
        "tests.test_pipeline_integration",
        "tests.test_e2e"
    ]
    
    results = []
    total_tests = 0
    total_passed = 0
    total_failures = 0
    total_errors = 0
    
    for module in test_modules:
        print(f"\nTesting {module}...")
        result = run_test_module(module)
        results.append(result)
        
        total_tests += result["tests"]
        total_passed += result["passed"]
        total_failures += result["failures"]
        total_errors += result["errors"]
        
        status = "✓ PASSED" if result["success"] else "✗ FAILED"
        print(f"  {status}: {result['tests']} tests, {result['passed']} passed, "
              f"{result['failures']} failures, {result['errors']} errors")
    
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)")
    print(f"Failed: {total_failures}")
    print(f"Errors: {total_errors}")
    print("\nDetailed Results:")
    
    for result in results:
        print(f"\n{result['module']}:")
        print(f"  Tests: {result['tests']}")
        print(f"  Success Rate: {result['passed']/result['tests']*100:.1f}% " if result['tests'] > 0 else "  No tests")
        if not result['success']:
            print(f"  Issues: {result['failures']} failures, {result['errors']} errors")
    
    # Overall status
    print("\n" + "=" * 60)
    if total_failures == 0 and total_errors == 0:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED - Further debugging needed")
        print("\nRecommended actions:")
        print("1. Fix failing unit tests first (especially app and detection modules)")
        print("2. Update integration tests to match current implementation")
        print("3. Ensure all dependencies are properly mocked")
        print("4. Check file paths and configurations in tests")

if __name__ == "__main__":
    main()