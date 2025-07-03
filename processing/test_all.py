#!/usr/bin/env python3
"""
Comprehensive Test Suite for Defect Detector
Runs all unit, integration, and end-to-end tests
"""

import os
import sys
import unittest
import logging
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test discovery patterns
TEST_PATTERNS = {
    'unit': 'test_*_unit.py',
    'integration': 'test_*_integration.py',
    'e2e': 'test_*_e2e.py'
}

class TestRunner:
    """Main test runner orchestrator"""
    
    def __init__(self):
        self.results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'test_suites': {}
        }
        self.start_time = None
        self.test_dir = Path(__file__).parent / "tests"
        
    def setup_test_environment(self):
        """Create necessary test directories and files"""
        # Create tests directory
        self.test_dir.mkdir(exist_ok=True)
        
        # Create test data directory
        test_data_dir = self.test_dir / "test_data"
        test_data_dir.mkdir(exist_ok=True)
        
        # Create test images directory
        test_images_dir = test_data_dir / "test_images"
        test_images_dir.mkdir(exist_ok=True)
        
        # Create test outputs directory
        test_outputs_dir = self.test_dir / "test_outputs"
        test_outputs_dir.mkdir(exist_ok=True)
        
        # Create __init__.py files
        (self.test_dir / "__init__.py").touch()
        
        logging.info(f"Test environment setup complete at {self.test_dir}")
        
    def discover_tests(self, pattern):
        """Discover tests matching the given pattern"""
        if not self.test_dir.exists():
            logging.warning(f"Test directory {self.test_dir} does not exist")
            return unittest.TestSuite()
            
        loader = unittest.TestLoader()
        suite = loader.discover(
            start_dir=str(self.test_dir),
            pattern=pattern,
            top_level_dir=str(Path(__file__).parent)
        )
        return suite
    
    def run_test_suite(self, suite_name, pattern):
        """Run a specific test suite"""
        logging.info(f"\n{'='*60}")
        logging.info(f"Running {suite_name} tests...")
        logging.info(f"{'='*60}")
        
        suite = self.discover_tests(pattern)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Store results
        self.results['test_suites'][suite_name] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0
        }
        
        self.results['total_tests'] += result.testsRun
        self.results['failed'] += len(result.failures)
        self.results['errors'] += len(result.errors)
        
        if hasattr(result, 'skipped'):
            self.results['skipped'] += len(result.skipped)
            
        self.results['passed'] = self.results['total_tests'] - \
                                self.results['failed'] - \
                                self.results['errors'] - \
                                self.results['skipped']
        
        return result.wasSuccessful()
    
    def run_all_tests(self):
        """Run all test suites"""
        self.start_time = datetime.now()
        
        # Setup test environment
        self.setup_test_environment()
        
        # Run each test suite
        all_successful = True
        for suite_name, pattern in TEST_PATTERNS.items():
            success = self.run_test_suite(suite_name, pattern)
            all_successful = all_successful and success
        
        # Calculate duration
        duration = (datetime.now() - self.start_time).total_seconds()
        self.results['duration'] = duration
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
        
        return all_successful
    
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Total Tests Run: {self.results['total_tests']}")
        print(f"Passed: {self.results['passed']} ✓")
        print(f"Failed: {self.results['failed']} ✗")
        print(f"Errors: {self.results['errors']} ⚠")
        print(f"Skipped: {self.results['skipped']} ⏭")
        print(f"Duration: {self.results['duration']:.2f} seconds")
        print("\nTest Suites:")
        for suite, stats in self.results['test_suites'].items():
            print(f"  {suite}: {stats['tests_run']} tests, "
                  f"{stats['failures']} failures, "
                  f"{stats['errors']} errors")
        print("="*60)
        
        # Print overall status
        if self.results['failed'] == 0 and self.results['errors'] == 0:
            print("✓ ALL TESTS PASSED!")
        else:
            print("✗ SOME TESTS FAILED!")
    
    def save_results(self):
        """Save test results to JSON file"""
        results_file = self.test_dir / "test_results.json"
        self.results['timestamp'] = datetime.now().isoformat()
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logging.info(f"Test results saved to {results_file}")

def main():
    """Main entry point"""
    runner = TestRunner()
    success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()