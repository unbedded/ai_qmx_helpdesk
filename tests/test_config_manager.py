"""
test_config_manager.py - Unit tests for ConfigManager module
<DATE>: 2025-08-25

Comprehensive test suite for the ConfigManager class ensuring thread-safety,
error handling, and proper configuration management functionality.

Example usage:
    pytest tests/test_config_manager.py -v
    pytest tests/test_config_manager.py::TestConfigManager::test_initialization
"""

import pytest
import threading
import time
from unittest.mock import patch, MagicMock

from ai_qmx_helpdesk.config_manager import ConfigManager, DEFAULT_CONFIG


class TestConfigManager:
    """Test suite for ConfigManager class."""

    def test_initialization_default(self) -> None:
        """Test initialization with default configuration."""
        config = ConfigManager()

        assert isinstance(config.get_cfg(), dict)
        assert len(config.get_cfg()) == len(DEFAULT_CONFIG)

        for key, value in DEFAULT_CONFIG.items():
            assert config.get_value(key) == value

    def test_initialization_with_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        custom_config = {"debug_mode": True, "timeout": 60, "custom_param": "test_value"}

        config = ConfigManager(cfg_dict=custom_config)

        # Check that custom values override defaults
        assert config.get_value("debug_mode") is True
        assert config.get_value("timeout") == 60
        assert config.get_value("custom_param") == "test_value"

        # Check that default values are still present
        assert config.get_value("log_level") == DEFAULT_CONFIG["log_level"]
        assert config.get_value("max_retries") == DEFAULT_CONFIG["max_retries"]

    def test_initialization_with_invalid_config(self) -> None:
        """Test initialization with invalid configuration type."""
        with pytest.raises(TypeError, match="cfg_dict must be a dictionary"):
            ConfigManager(cfg_dict="invalid")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="cfg_dict must be a dictionary"):
            ConfigManager(cfg_dict=123)  # type: ignore[arg-type]

    def test_get_cfg(self) -> None:
        """Test getting full configuration dictionary."""
        custom_config = {"test_key": "test_value"}
        config = ConfigManager(cfg_dict=custom_config)

        cfg = config.get_cfg()

        assert isinstance(cfg, dict)
        assert "test_key" in cfg
        assert cfg["test_key"] == "test_value"

        # Ensure it's a copy, not reference
        cfg["new_key"] = "new_value"
        assert "new_key" not in config.get_cfg()

    def test_set_cfg_valid(self) -> None:
        """Test setting configuration with valid dictionary."""
        config = ConfigManager()

        new_config = {"debug_mode": True, "timeout": 45, "new_param": "new_value"}

        config.set_cfg(new_config)

        assert config.get_value("debug_mode") is True
        assert config.get_value("timeout") == 45
        # new_param should not be set since it doesn't exist in original config
        assert config.get_value("new_param") is None

    def test_set_cfg_invalid(self) -> None:
        """Test setting configuration with invalid inputs."""
        config = ConfigManager()

        with pytest.raises(TypeError, match="cfg_dict must be a dictionary"):
            config.set_cfg("invalid")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="cfg_dict cannot be empty"):
            config.set_cfg({})

    def test_get_value(self) -> None:
        """Test getting individual configuration values."""
        config = ConfigManager({"test_key": "test_value"})

        # Test existing key
        assert config.get_value("test_key") == "test_value"
        assert config.get_value("log_level") == DEFAULT_CONFIG["log_level"]

        # Test non-existing key with default
        assert config.get_value("non_existing") is None
        assert config.get_value("non_existing", "default") == "default"

    def test_get_value_invalid_key(self) -> None:
        """Test getting value with invalid key type."""
        config = ConfigManager()

        with pytest.raises(TypeError, match="Key must be a string"):
            config.get_value(123)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Key must be a string"):
            config.get_value(None)  # type: ignore[arg-type]

    def test_set_value(self) -> None:
        """Test setting individual configuration values."""
        config = ConfigManager()

        config.set_value("test_key", "test_value")
        assert config.get_value("test_key") == "test_value"

        # Test updating existing value
        config.set_value("debug_mode", True)
        assert config.get_value("debug_mode") is True

    def test_set_value_invalid_key(self) -> None:
        """Test setting value with invalid key type."""
        config = ConfigManager()

        with pytest.raises(TypeError, match="Key must be a string"):
            config.set_value(123, "value")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Key must be a string"):
            config.set_value(None, "value")  # type: ignore[arg-type]

    def test_has_key(self) -> None:
        """Test checking if configuration key exists."""
        config = ConfigManager({"test_key": "test_value"})

        assert config.has_key("test_key") is True
        assert config.has_key("log_level") is True
        assert config.has_key("non_existing") is False

    def test_has_key_invalid(self) -> None:
        """Test has_key with invalid key type."""
        config = ConfigManager()

        with pytest.raises(TypeError, match="Key must be a string"):
            config.has_key(123)  # type: ignore[arg-type]

    def test_remove_key(self) -> None:
        """Test removing configuration keys."""
        config = ConfigManager({"test_key": "test_value"})

        # Test removing existing key
        assert config.remove_key("test_key") is True
        assert config.has_key("test_key") is False

        # Test removing non-existing key
        assert config.remove_key("non_existing") is False

    def test_remove_key_invalid(self) -> None:
        """Test remove_key with invalid key type."""
        config = ConfigManager()

        with pytest.raises(TypeError, match="Key must be a string"):
            config.remove_key(123)  # type: ignore[arg-type]

    def test_get_keys(self) -> None:
        """Test getting list of all configuration keys."""
        custom_config = {"test_key": "test_value"}
        config = ConfigManager(cfg_dict=custom_config)

        keys = config.get_keys()

        assert isinstance(keys, list)
        assert "test_key" in keys
        assert all(key in keys for key in DEFAULT_CONFIG.keys())

    def test_clear(self) -> None:
        """Test clearing configuration and restoring defaults."""
        custom_config = {"test_key": "test_value", "debug_mode": True}
        config = ConfigManager(cfg_dict=custom_config)

        config.clear()

        # Should have only default keys
        assert len(config.get_keys()) == len(DEFAULT_CONFIG)
        assert config.get_value("test_key") is None
        assert config.get_value("debug_mode") == DEFAULT_CONFIG["debug_mode"]

    def test_thread_safety(self) -> None:
        """Test thread-safe operations on configuration."""
        config = ConfigManager()
        results = []
        errors = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(100):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"value_{i}"

                    config.set_value(key, value)
                    retrieved = config.get_value(key)

                    if retrieved == value:
                        results.append(f"{worker_id}:{i}")

                    time.sleep(0.001)  # Small delay to increase chance of race conditions
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 500  # 5 workers * 100 operations each

    def test_logging_configuration(self) -> None:
        """Test that logging is properly configured."""
        with patch("logging.getLogger") as mock_logger:
            mock_logger_instance = MagicMock()
            mock_logger.return_value = mock_logger_instance

            ConfigManager()

            # Verify logger was created
            mock_logger.assert_called_once()
            mock_logger_instance.setLevel.assert_called_once()

    def test_string_representations(self) -> None:
        """Test __str__ and __repr__ methods."""
        config = ConfigManager({"test_key": "test_value"})

        str_repr = str(config)
        assert "ConfigManager" in str_repr
        assert "parameters" in str_repr

        repr_str = repr(config)
        assert "ConfigManager" in repr_str
        assert "cfg_dict" in repr_str

    def test_edge_cases(self) -> None:
        """Test various edge cases."""
        config = ConfigManager()

        # Test empty string key
        config.set_value("", "empty_key_value")
        assert config.get_value("") == "empty_key_value"

        # Test None value
        config.set_value("none_value", None)
        assert config.get_value("none_value") is None

        # Test complex data types
        complex_value = {"nested": {"data": [1, 2, 3]}}
        config.set_value("complex", complex_value)
        assert config.get_value("complex") == complex_value


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager."""

    def test_full_workflow(self) -> None:
        """Test complete configuration management workflow."""
        # Initialize with custom config
        initial_config = {"app_name": "test_app", "version": "1.0.0", "debug_mode": True}

        config = ConfigManager(cfg_dict=initial_config)

        # Verify initial state
        assert config.get_value("app_name") == "test_app"
        assert config.get_value("debug_mode") is True

        # Update configuration
        updates = {"debug_mode": False, "timeout": 120}
        config.set_cfg(updates)

        # Verify updates
        assert config.get_value("debug_mode") is False
        assert config.get_value("timeout") == 120

        # Add new values
        config.set_value("new_feature", True)
        assert config.get_value("new_feature") is True

        # Remove values
        assert config.remove_key("new_feature") is True
        assert config.get_value("new_feature") is None

        # Clear and verify defaults
        config.clear()
        assert config.get_value("app_name") is None
        assert config.get_value("debug_mode") == DEFAULT_CONFIG["debug_mode"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
