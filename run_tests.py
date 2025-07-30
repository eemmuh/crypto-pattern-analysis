#!/usr/bin/env python3
"""
Test runner script for crypto trading analysis project.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for crypto trading analysis")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "all", "coverage", "performance"],
        default="unit",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--html", 
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Skip slow tests"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        cmd.append("-v")
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    if args.fast:
        cmd.append("-m")
        cmd.append("not slow")
    
    # Test type specific commands
    if args.type == "unit":
        cmd.extend([
            "tests/unit/",
            "-m", "unit"
        ])
    elif args.type == "integration":
        cmd.extend([
            "tests/integration/",
            "-m", "integration"
        ])
    elif args.type == "coverage":
        cmd.extend([
            "tests/",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-fail-under=80"
        ])
    elif args.type == "performance":
        cmd.extend([
            "tests/",
            "-m", "performance",
            "--benchmark-only"
        ])
    elif args.type == "all":
        cmd.extend([
            "tests/",
            "--cov=src",
            "--cov-report=term-missing"
        ])
        if args.html:
            cmd.append("--cov-report=html:htmlcov")
    
    # Run the tests
    success = run_command(cmd, f"Running {args.type} tests")
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 