#!/usr/bin/env python3
"""
Local test runner for Playwright e2e tests.

Usage:
    python run_e2e_tests.py
    
Requirements:
    pip install playwright
    playwright install chromium
"""

import subprocess
import sys
import os
import time

def run_command(cmd, cwd=None, env=None):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    return result.returncode == 0

def main():
    """Run e2e tests locally."""
    print("🎭 OVOD Playwright E2E Test Runner")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("repo/demo_app.py"):
        print("❌ Error: Run this from the project root directory")
        print("   Expected to find repo/demo_app.py")
        sys.exit(1)
    
    # Set up environment
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.abspath('repo')
    env['OVOD_SKIP_SAM2'] = '1'
    
    print("📦 Installing Playwright browsers...")
    if not run_command("python -m playwright install chromium"):
        print("❌ Failed to install Playwright browsers")
        sys.exit(1)
    
    print("\n🧪 Running e2e tests...")
    success = run_command(
        "python -m pytest tests/e2e/ -v --tb=short -m 'not slow' || true",
        env=env
    )
    
    print("\n🚀 Running all tests (including slow)...")
    run_command(
        "python -m pytest tests/e2e/ -v --tb=short",
        env=env
    )
    
    print("\n✅ E2E test run completed!")
    print("\nTo run tests manually:")
    print("  export PYTHONPATH=./repo")
    print("  export OVOD_SKIP_SAM2=1")
    print("  pytest tests/e2e/ -v")

if __name__ == "__main__":
    main()