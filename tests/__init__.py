"""
Test package for crypto trading analysis project.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Test configuration
TEST_CONFIG = {
    'test_data_dir': 'tests/test_data',
    'cache_dir': 'tests/test_cache',
    'log_level': 'DEBUG'
}

# Ensure test directories exist
os.makedirs(TEST_CONFIG['test_data_dir'], exist_ok=True)
os.makedirs(TEST_CONFIG['cache_dir'], exist_ok=True) 