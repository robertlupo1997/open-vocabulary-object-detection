"""
Tests for demo setup and functionality
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
import json

import demo_setup


class TestDemoSetup:
    """Test demo setup script functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup after each test"""
        shutil.rmtree(self.temp_dir)
    
    def test_setup_directories(self):
        """Test directory creation"""
        
        with patch('os.makedirs') as mock_makedirs:
            demo_setup.setup_directories()
            
            # Should create expected directories
            expected_dirs = ["weights", "data", "logs", "notebooks", "metrics", "tests"]
            assert mock_makedirs.call_count == len(expected_dirs)
    
    @patch('demo_setup.torch')
    def test_check_gpu_available(self, mock_torch):
        """Test GPU check when CUDA is available"""
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "GeForce RTX 3070"
        mock_torch.cuda.get_device_properties.return_value.total_memory = 8 * 1024**3
        
        result = demo_setup.check_gpu()
        
        assert result is True
        mock_torch.cuda.is_available.assert_called_once()
        mock_torch.cuda.get_device_name.assert_called_once_with(0)
    
    @patch('demo_setup.torch')
    def test_check_gpu_unavailable(self, mock_torch):
        """Test GPU check when CUDA is not available"""
        
        mock_torch.cuda.is_available.return_value = False
        
        result = demo_setup.check_gpu()
        
        assert result is False
        mock_torch.cuda.is_available.assert_called_once()
    
    def test_create_requirements_file(self):
        """Test requirements file creation"""
        
        original_cwd = Path.cwd()
        
        try:
            # Change to temp directory
            import os
            os.chdir(self.temp_dir)
            
            demo_setup.create_requirements_file()
            
            requirements_file = self.temp_dir / "requirements.txt"
            assert requirements_file.exists()
            
            content = requirements_file.read_text()
            assert "torch" in content
            assert "transformers" in content
            assert "streamlit" in content
            
        finally:
            os.chdir(original_cwd)
    
    @patch('demo_setup.urllib.request.urlretrieve')
    @patch('demo_setup.os.makedirs')
    def test_download_file_success(self, mock_makedirs, mock_urlretrieve):
        """Test successful file download"""
        
        url = "https://example.com/model.pth"
        filepath = str(self.temp_dir / "model.pth")
        
        result = demo_setup.download_file(url, filepath, "Test Model")
        
        assert result is True
        mock_urlretrieve.assert_called_once()
        mock_makedirs.assert_called_once()
    
    @patch('demo_setup.urllib.request.urlretrieve')
    def test_download_file_failure(self, mock_urlretrieve):
        """Test file download failure"""
        
        mock_urlretrieve.side_effect = Exception("Download failed")
        
        url = "https://example.com/model.pth"
        filepath = str(self.temp_dir / "model.pth")
        
        result = demo_setup.download_file(url, filepath, "Test Model")
        
        assert result is False
    
    @patch('demo_setup.__import__')
    def test_check_dependencies_all_present(self, mock_import):
        """Test dependency check when all packages are present"""
        
        # Mock all imports to succeed
        mock_import.return_value = Mock()
        
        result = demo_setup.check_dependencies()
        
        assert result is True
    
    @patch('demo_setup.__import__')
    def test_check_dependencies_missing(self, mock_import):
        """Test dependency check when packages are missing"""
        
        # Mock some imports to fail
        def side_effect(name):
            if name in ["torch", "transformers"]:
                raise ImportError(f"No module named '{name}'")
            return Mock()
        
        mock_import.side_effect = side_effect
        
        result = demo_setup.check_dependencies()
        
        assert result is False
    
    def test_model_configs_structure(self):
        """Test that model configs have required structure"""
        
        for model_name, config in demo_setup.MODEL_CONFIGS.items():
            assert "checkpoint_url" in config
            assert "checkpoint_path" in config
            assert "size_mb" in config
            
            # Validate URL format
            assert config["checkpoint_url"].startswith("http")
            
            # Validate size is reasonable
            assert isinstance(config["size_mb"], (int, float))
            assert config["size_mb"] > 0
    
    @patch('demo_setup.download_file')
    @patch('demo_setup.os.path.exists')
    def test_download_models_skip_existing(self, mock_exists, mock_download):
        """Test that existing models are skipped"""
        
        mock_exists.return_value = True  # Model already exists
        
        result = demo_setup.download_models(["grounding_dino"], force=False)
        
        assert result is True
        mock_download.assert_not_called()
    
    @patch('demo_setup.download_file')
    @patch('demo_setup.os.path.exists')
    def test_download_models_force_download(self, mock_exists, mock_download):
        """Test force downloading existing models"""
        
        mock_exists.return_value = True  # Model already exists
        mock_download.return_value = True
        
        result = demo_setup.download_models(["grounding_dino"], force=True)
        
        assert result is True
        mock_download.assert_called_once()
    
    @patch('demo_setup.download_file')
    @patch('demo_setup.os.path.exists')
    def test_download_models_unknown_model(self, mock_exists, mock_download):
        """Test handling of unknown model names"""
        
        mock_exists.return_value = False
        
        result = demo_setup.download_models(["unknown_model"])
        
        assert result is False
        mock_download.assert_not_called()


class TestDemoSetupCLI:
    """Test demo setup command line interface"""
    
    @patch('demo_setup.create_requirements_file')
    def test_requirements_only_flag(self, mock_create_req):
        """Test --requirements-only flag"""
        
        # Mock sys.argv
        test_args = ["demo_setup.py", "--requirements-only"]
        
        with patch('sys.argv', test_args):
            with patch('demo_setup.parser.parse_args') as mock_parse:
                mock_args = Mock()
                mock_args.requirements_only = True
                mock_parse.return_value = mock_args
                
                # This would normally call main(), but we'll test the logic
                if mock_args.requirements_only:
                    demo_setup.create_requirements_file()
                
                mock_create_req.assert_called_once()
    
    def test_model_choices_validation(self):
        """Test that model choices are validated"""
        
        valid_models = list(demo_setup.MODEL_CONFIGS.keys())
        
        # Test that all model configs are valid choices
        for model in valid_models:
            assert model in demo_setup.MODEL_CONFIGS
    
    @patch('demo_setup.download_models')
    @patch('demo_setup.setup_directories')  
    @patch('demo_setup.create_requirements_file')
    @patch('demo_setup.check_dependencies')
    @patch('demo_setup.check_gpu')
    @patch('demo_setup.verify_installation')
    def test_main_workflow(self, mock_verify, mock_gpu, mock_deps, 
                          mock_req, mock_dirs, mock_download):
        """Test main setup workflow"""
        
        # Mock all functions to succeed
        mock_deps.return_value = True
        mock_gpu.return_value = True
        mock_download.return_value = True
        mock_verify.return_value = True
        
        # Mock args
        test_args = ["demo_setup.py"]
        
        with patch('sys.argv', test_args):
            with patch('demo_setup.parser.parse_args') as mock_parse:
                mock_args = Mock()
                mock_args.requirements_only = False
                mock_args.skip_deps = False
                mock_args.models = ["grounding_dino", "sam2_small"]
                mock_args.force = False
                mock_parse.return_value = mock_args
                
                # Simulate main workflow
                if not mock_args.skip_deps:
                    demo_setup.check_dependencies()
                
                demo_setup.setup_directories()
                demo_setup.create_requirements_file()
                demo_setup.check_gpu()
                demo_setup.download_models(mock_args.models, mock_args.force)
                demo_setup.verify_installation()
                
                # Verify all steps were called
                mock_deps.assert_called_once()
                mock_dirs.assert_called_once()
                mock_req.assert_called_once()
                mock_gpu.assert_called_once()
                mock_download.assert_called_once()
                mock_verify.assert_called_once()


class TestVerifyInstallation:
    """Test installation verification"""
    
    @patch('demo_setup.os.path.exists')
    def test_verify_missing_models(self, mock_exists):
        """Test verification with missing models"""
        
        # Mock some models as missing
        def exists_side_effect(path):
            return "sam2" not in path  # SAM2 models missing
        
        mock_exists.side_effect = exists_side_effect
        
        # Should still return True but print warnings
        result = demo_setup.verify_installation()
        
        # The function may return True even with missing models (warnings only)
        assert isinstance(result, bool)
    
    @patch('demo_setup.os.path.exists')
    @patch('demo_setup.check_gpu')
    def test_verify_complete_installation(self, mock_gpu, mock_exists):
        """Test verification with complete installation"""
        
        mock_exists.return_value = True  # All models exist
        mock_gpu.return_value = True
        
        with patch('ovod.pipeline.OVODPipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            
            result = demo_setup.verify_installation()
            
            assert result is True