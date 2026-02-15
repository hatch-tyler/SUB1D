"""Tests for configuration loading and validation."""
from __future__ import annotations

import os
import tempfile

import pytest

from sub1d.config import (
    _str_to_bool,
    load_par_file,
    ModelConfig,
    AdminConfig,
    validate_config,
)
from sub1d.exceptions import ConfigurationError


class TestStrToBool:
    """Test the strtobool replacement."""

    def test_true_values(self):
        for val in ["true", "True", "TRUE", "1", "yes", "Yes", "on", "ON"]:
            assert _str_to_bool(val) is True

    def test_false_values(self):
        for val in ["false", "False", "FALSE", "0", "no", "No", "off", "OFF"]:
            assert _str_to_bool(val) is False

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _str_to_bool("maybe")

    def test_whitespace_handling(self):
        assert _str_to_bool("  true  ") is True
        assert _str_to_bool(" 0 ") is False


class TestParFileLoading:
    """Test loading legacy .par files."""

    def test_loads_basic_par_file(self, sample_par_content):
        """Should load a well-formed .par file without error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".par",
                                          delete=False) as f:
            f.write(sample_par_content)
            f.flush()
            par_path = f.name

        try:
            config = load_par_file(par_path)
            assert isinstance(config, ModelConfig)
            assert config.admin.run_name == "test_run"
            assert config.admin.output_folder == "Output/"
            assert len(config.layer_names) == 3
            assert config.layer_types["Upper Aquifer"] == "Aquifer"
            assert config.layer_types["Clay Layer"] == "Aquitard"
        finally:
            os.unlink(par_path)

    def test_reads_layer_count(self, sample_par_content):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".par",
                                          delete=False) as f:
            f.write(sample_par_content)
            f.flush()
            par_path = f.name

        try:
            config = load_par_file(par_path)
            assert len(config.layers) == 3
        finally:
            os.unlink(par_path)

    def test_reads_hydrologic_params(self, sample_par_content):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".par",
                                          delete=False) as f:
            f.write(sample_par_content)
            f.flush()
            par_path = f.name

        try:
            config = load_par_file(par_path)
            assert config.hydro.rho_w == 1000.0
            assert config.hydro.g == 9.81
            assert config.hydro.compressibility_of_water == 1.5e-6
        finally:
            os.unlink(par_path)


class TestValidation:
    """Test configuration validation."""

    def test_validate_passes_for_valid_config(self, sample_par_content):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".par",
                                          delete=False) as f:
            f.write(sample_par_content)
            f.flush()
            par_path = f.name

        try:
            config = load_par_file(par_path)
            # Should not raise
            validate_config(config)
        finally:
            os.unlink(par_path)
