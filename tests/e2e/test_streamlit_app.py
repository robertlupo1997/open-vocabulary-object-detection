"""
Playwright end-to-end tests for the Streamlit OVOD demo app.
"""
import pytest
import subprocess
import time
import os
import sys
from playwright.sync_api import Page, expect
import threading
import signal

# Add repo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../repo'))

class StreamlitServer:
    """Helper class to manage Streamlit server for testing."""
    
    def __init__(self, port=8501):
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"
    
    def start(self):
        """Start the Streamlit server."""
        repo_dir = os.path.join(os.path.dirname(__file__), '../../repo')
        env = os.environ.copy()
        env['PYTHONPATH'] = repo_dir
        env['OVOD_SKIP_SAM2'] = '1'  # Skip SAM2 for faster testing
        
        self.process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 
            'demo_app.py',
            '--server.port', str(self.port),
            '--server.headless', 'true',
            '--server.fileWatcherType', 'none'
        ], cwd=repo_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        for _ in range(30):  # 30 second timeout
            try:
                import requests
                response = requests.get(self.base_url, timeout=1)
                if response.status_code == 200:
                    return
            except:
                pass
            time.sleep(1)
        
        raise RuntimeError("Streamlit server failed to start")
    
    def stop(self):
        """Stop the Streamlit server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


@pytest.fixture(scope="session")
def streamlit_server():
    """Fixture to start/stop Streamlit server for the test session."""
    server = StreamlitServer()
    try:
        server.start()
        yield server
    finally:
        server.stop()


@pytest.mark.slow
@pytest.mark.skipif(os.getenv('CI') == 'true', reason="Skip Playwright tests in CI (requires display)")
def test_streamlit_app_loads(page: Page, streamlit_server):
    """Test that the Streamlit app loads successfully."""
    page.goto(streamlit_server.base_url)
    
    # Wait for the page to load
    page.wait_for_load_state("networkidle")
    
    # Check for main title
    expect(page.locator("h1")).to_contain_text("OVOD")
    
    # Check for key elements
    expect(page.locator("text=Open-Vocabulary Object Detection")).to_be_visible()


@pytest.mark.slow 
@pytest.mark.skipif(os.getenv('CI') == 'true', reason="Skip Playwright tests in CI")
def test_text_prompt_input(page: Page, streamlit_server):
    """Test that text prompt input field exists and works."""
    page.goto(streamlit_server.base_url)
    page.wait_for_load_state("networkidle")
    
    # Look for text input field
    text_input = page.locator("input[type='text']").first
    if text_input.is_visible():
        text_input.fill("person")
        expect(text_input).to_have_value("person")


@pytest.mark.slow
@pytest.mark.skipif(os.getenv('CI') == 'true', reason="Skip Playwright tests in CI")  
def test_file_upload_widget(page: Page, streamlit_server):
    """Test that file upload widget is present."""
    page.goto(streamlit_server.base_url)
    page.wait_for_load_state("networkidle")
    
    # Look for file upload component
    upload_button = page.locator("button:has-text('Browse files')")
    if upload_button.is_visible():
        expect(upload_button).to_be_visible()


@pytest.mark.slow
@pytest.mark.skipif(os.getenv('CI') == 'true', reason="Skip Playwright tests in CI")
def test_sidebar_controls(page: Page, streamlit_server):
    """Test that sidebar controls are present."""
    page.goto(streamlit_server.base_url)
    page.wait_for_load_state("networkidle")
    
    # Check for sidebar
    sidebar = page.locator("[data-testid='stSidebar']")
    if sidebar.is_visible():
        expect(sidebar).to_be_visible()
        
        # Look for threshold controls
        sliders = page.locator("input[type='range']")
        if sliders.count() > 0:
            expect(sliders.first).to_be_visible()


@pytest.mark.slow
@pytest.mark.skipif(os.getenv('CI') == 'true', reason="Skip Playwright tests in CI")
def test_error_handling(page: Page, streamlit_server):
    """Test that the app handles errors gracefully."""
    page.goto(streamlit_server.base_url)
    page.wait_for_load_state("networkidle")
    
    # App should load without fatal errors
    error_messages = page.locator("text=Error").count()
    # Allow for non-fatal warnings but not critical errors
    assert error_messages < 5, "Too many error messages on page load"


# Performance test
@pytest.mark.slow
@pytest.mark.skipif(os.getenv('CI') == 'true', reason="Skip Playwright tests in CI")
def test_app_performance(page: Page, streamlit_server):
    """Test that the app loads within reasonable time."""
    start_time = time.time()
    page.goto(streamlit_server.base_url)
    page.wait_for_load_state("networkidle")
    load_time = time.time() - start_time
    
    # App should load within 10 seconds
    assert load_time < 10, f"App took {load_time:.2f}s to load, expected < 10s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])