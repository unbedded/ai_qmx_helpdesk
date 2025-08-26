"""
config_manager.py - Configuration Management Module
<DATE>: 2025-08-25

This module provides a centralized configuration management system with
support for default values, validation, and thread-safe operations.

Example usage:
    from config_manager import ConfigManager

    # Initialize with default configuration
    config = ConfigManager()

    # Initialize with custom configuration
    custom_config = {
        'database_url': 'postgresql://localhost:5432/test',
        'debug_mode': True,
        'max_connections': 10
    }
    config = ConfigManager(cfg_dict=custom_config)

    # Get configuration value
    db_url = config.get_value('database_url')

    # Set configuration value
    config.set_value('timeout', 30)

    # Get full configuration
    full_config = config.get_cfg()

    # Update configuration
    config.set_cfg({'debug_mode': False, 'timeout': 60})
"""

import logging
import threading
from typing import Dict, Any, List, Optional


# STEP_1: Define configuration constants
DEFAULT_CONFIG: Dict[str, Any] = {
    "log_level": "WARNING",
    "max_retries": 3,
    "timeout": 30,
    "debug_mode": False,
    "cache_size": 100,
    "thread_pool_size": 4,
}

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class ConfigManager:
    """
    A thread-safe configuration manager that handles application settings
    with default values, validation, and logging.

    This class provides methods to get, set, and manage configuration
    parameters with built-in error handling and logging capabilities.

    Attributes:
        _cfg_dict (Dict[str, Any]): Internal configuration dictionary
        _lock (threading.RLock): Thread synchronization lock
        _logger (logging.Logger): Logger instance for this class
    """

    def __init__(self, cfg_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the ConfigManager with optional configuration dictionary.

        STEP_2: Initialize logging as first step in constructor
        STEP_3: Initialize configuration with defaults and user values
        STEP_4: Set up thread-safe operations with lock

        Args:
            cfg_dict: Optional dictionary of configuration parameters.
                     If None, uses empty dict and applies defaults.

        Raises:
            TypeError: If cfg_dict is not a dictionary or None
        """
        # STEP_2: Initialize logging first
        self._logger = logging.getLogger(__name__ + ".ConfigManager")
        self._logger.setLevel(logging.WARNING)

        # Configure logging with file output and timestamps
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(LOG_FORMAT)
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

        self._logger.debug("Initializing ConfigManager")

        # STEP_3: Initialize configuration
        if cfg_dict is None:
            cfg_dict = {}

        if not isinstance(cfg_dict, dict):
            error_msg = f"cfg_dict must be a dictionary, got {type(cfg_dict)}"
            self._logger.error(error_msg)
            raise TypeError(error_msg)

        # STEP_4: Set up thread-safe operations
        self._lock = threading.RLock()

        with self._lock:
            self._cfg_dict = DEFAULT_CONFIG.copy()
            self._apply_user_config(cfg_dict)

        self._logger.info("ConfigManager initialized with %d parameters", len(self._cfg_dict))

    def _apply_user_config(self, user_config: Dict[str, Any]) -> None:
        """
        Apply user configuration, logging updates and mismatches.

        STEP_5: Validate and apply user configuration parameters

        Args:
            user_config: Dictionary of user-provided configuration
        """
        for key, value in user_config.items():
            if key in self._cfg_dict:
                old_value = self._cfg_dict[key]
                self._cfg_dict[key] = value
                self._logger.debug("Updated parameter '%s': %s -> %s", key, old_value, value)
            else:
                self._cfg_dict[key] = value
                self._logger.info("Added new parameter '%s': %s", key, value)

    def get_cfg(self) -> Dict[str, Any]:
        """
        Get a copy of the current configuration dictionary.

        STEP_6: Return thread-safe copy of configuration

        Returns:
            Dict[str, Any]: Copy of current configuration parameters
        """
        with self._lock:
            config_copy = self._cfg_dict.copy()
            self._logger.debug("Retrieved configuration with %d parameters", len(config_copy))
            return config_copy

    def set_cfg(self, cfg_dict: Dict[str, Any]) -> None:
        """
        Update configuration with new dictionary, logging all changes.

        STEP_7: Thread-safe configuration update with validation

        Args:
            cfg_dict: Dictionary of configuration parameters to update

        Raises:
            TypeError: If cfg_dict is not a dictionary
            ValueError: If cfg_dict is empty
        """
        if not isinstance(cfg_dict, dict):
            error_msg = f"cfg_dict must be a dictionary, got {type(cfg_dict)}"
            self._logger.error(error_msg)
            raise TypeError(error_msg)

        if not cfg_dict:
            error_msg = "cfg_dict cannot be empty"
            self._logger.error(error_msg)
            raise ValueError(error_msg)

        with self._lock:
            updated_keys = []
            mismatched_keys = []

            for key, value in cfg_dict.items():
                if key in self._cfg_dict:
                    old_value = self._cfg_dict[key]
                    self._cfg_dict[key] = value
                    updated_keys.append(key)
                    self._logger.debug("Updated parameter '%s': %s -> %s", key, old_value, value)
                else:
                    mismatched_keys.append(key)
                    self._logger.info(
                        "Key/parameter mismatch: '%s' not found in current config", key
                    )

            if updated_keys:
                self._logger.info("Updated parameters: %s", updated_keys)

            if mismatched_keys:
                self._logger.warning("Mismatched keys ignored: %s", mismatched_keys)

    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value with optional default.

        STEP_8: Thread-safe retrieval of individual configuration values

        Args:
            key: Configuration parameter key
            default: Default value if key not found

        Returns:
            Any: Configuration value or default
        """
        if not isinstance(key, str):
            error_msg = f"Key must be a string, got {type(key)}"
            self._logger.error(error_msg)
            raise TypeError(error_msg)

        with self._lock:
            if key in self._cfg_dict:
                value = self._cfg_dict[key]
                self._logger.debug("Retrieved value for '%s': %s", key, value)
                return value
            else:
                self._logger.debug("Key '%s' not found, returning default: %s", key, default)
                return default

    def set_value(self, key: str, value: Any) -> None:
        """
        Set a specific configuration value.

        STEP_9: Thread-safe setting of individual configuration values

        Args:
            key: Configuration parameter key
            value: Value to set

        Raises:
            TypeError: If key is not a string
        """
        if not isinstance(key, str):
            error_msg = f"Key must be a string, got {type(key)}"
            self._logger.error(error_msg)
            raise TypeError(error_msg)

        with self._lock:
            old_value = self._cfg_dict.get(key, "<not set>")
            self._cfg_dict[key] = value
            self._logger.debug("Set parameter '%s': %s -> %s", key, old_value, value)
            self._logger.info("Configuration parameter '%s' updated", key)

    def has_key(self, key: str) -> bool:
        """
        Check if a configuration key exists.

        STEP_10: Thread-safe key existence check

        Args:
            key: Configuration parameter key to check

        Returns:
            bool: True if key exists, False otherwise

        Raises:
            TypeError: If key is not a string
        """
        if not isinstance(key, str):
            error_msg = f"Key must be a string, got {type(key)}"
            self._logger.error(error_msg)
            raise TypeError(error_msg)

        with self._lock:
            exists = key in self._cfg_dict
            self._logger.debug("Key '%s' exists: %s", key, exists)
            return exists

    def remove_key(self, key: str) -> bool:
        """
        Remove a configuration key if it exists.

        STEP_11: Thread-safe key removal with logging

        Args:
            key: Configuration parameter key to remove

        Returns:
            bool: True if key was removed, False if key didn't exist

        Raises:
            TypeError: If key is not a string
        """
        if not isinstance(key, str):
            error_msg = f"Key must be a string, got {type(key)}"
            self._logger.error(error_msg)
            raise TypeError(error_msg)

        with self._lock:
            if key in self._cfg_dict:
                old_value = self._cfg_dict.pop(key)
                self._logger.info("Removed parameter '%s' with value: %s", key, old_value)
                return True
            else:
                self._logger.debug("Key '%s' not found for removal", key)
                return False

    def get_keys(self) -> List[str]:
        """
        Get list of all configuration keys.

        STEP_12: Thread-safe retrieval of all configuration keys

        Returns:
            List[str]: List of all configuration parameter keys
        """
        with self._lock:
            keys = list(self._cfg_dict.keys())
            self._logger.debug("Retrieved %d configuration keys", len(keys))
            return keys

    def clear(self) -> None:
        """
        Clear all configuration parameters and restore defaults.

        STEP_13: Thread-safe configuration reset to defaults
        """
        with self._lock:
            old_count = len(self._cfg_dict)
            self._cfg_dict.clear()
            self._cfg_dict.update(DEFAULT_CONFIG)
            self._logger.info(
                "Cleared %d parameters and restored %d defaults", old_count, len(DEFAULT_CONFIG)
            )

    def __str__(self) -> str:
        """String representation of the configuration."""
        with self._lock:
            return f"ConfigManager({len(self._cfg_dict)} parameters)"

    def __repr__(self) -> str:
        """Detailed string representation of the configuration."""
        with self._lock:
            return f"ConfigManager(cfg_dict={self._cfg_dict})"
