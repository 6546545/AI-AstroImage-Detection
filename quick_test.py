#!/usr/bin/env python3
"""
Quick test to check current success rate and identify issues.
"""

import unittest
import sys
import os

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def run_quick_test():
    """Run a quick test to check success rate."""
    try:
        from test_cosmosai_systems import TestCosmosAISystems
        
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(TestCosmosAISystems)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        # Calculate success rate
        total_tests = result.testsRun
        failed_tests = len(result.failures) + len(result.errors)
        success_rate = ((total_tests - failed_tests) / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 Quick Test Results:")
        print(f"   Total tests: {total_tests}")
        print(f"   Failed tests: {failed_tests}")
        print(f"   Success rate: {success_rate:.1f}%")
        
        if result.failures:
            print(f"\n❌ Failures ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"   - {test}")
        
        if result.errors:
            print(f"\n❌ Errors ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"   - {test}")
        
        return success_rate >= 90.0
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_quick_test()
    sys.exit(0 if success else 1)
