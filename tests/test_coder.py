"""
Tests for the MedicalCoder class
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.coder import MedicalCoder


class TestMedicalCoder:
    """Test suite for MedicalCoder class"""
    
    def test_initialization(self):
        """Test that MedicalCoder can be initialized"""
        coder = MedicalCoder()
        assert coder is not None
        assert isinstance(coder.config, dict)
    
    def test_initialization_with_config(self):
        """Test initialization with custom config"""
        config = {"test_key": "test_value"}
        coder = MedicalCoder(config=config)
        assert coder.config == config
    
    def test_suggest_codes_not_implemented(self):
        """Test that suggest_codes raises NotImplementedError"""
        coder = MedicalCoder()
        with pytest.raises(NotImplementedError):
            coder.suggest_codes("Sample clinical text")
    
    def test_validate_codes_not_implemented(self):
        """Test that validate_codes raises NotImplementedError"""
        coder = MedicalCoder()
        with pytest.raises(NotImplementedError):
            coder.validate_codes(["J02.9", "R50.9"])
