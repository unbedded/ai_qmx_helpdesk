"""
main.py - Main entry point for QMX Helpdesk application
<DATE>: 2025-08-25

This module provides the main entry point for the application, demonstrating
the usage of ConfigManager for managing application settings.

Example usage:
    python -m ai_qmx_helpdesk.main
    python src/ai_qmx_helpdesk/main.py

    # With custom config file:
    python -m ai_qmx_helpdesk.main --name "Alice"
"""

import argparse
import logging
from typing import Dict, Any

# from ai_qmx_helpdesk.config_manager import ConfigManager
from config_manager import ConfigManager


# STEP_1: Define application constants
DEFAULT_NAME = "Spencer"
APP_NAME = "QMX Helpdesk"


def setup_logging() -> None:
    """
    Configure application logging.

    STEP_2: Set up basic logging configuration for the application
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    STEP_3: Parse command line arguments for configuration overrides

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} - Configuration Management Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Use default name 'Spencer'
  %(prog)s --name Alice       # Use custom name 'Alice'
  %(prog)s --name "John Doe"  # Use name with spaces
        """,
    )

    parser.add_argument(
        "--name", type=str, default=DEFAULT_NAME, help=f"Name to greet (default: {DEFAULT_NAME})"
    )

    parser.add_argument("--config-file", type=str, help="Path to configuration file (optional)")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def load_config_from_file(config_file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file.

    STEP_4: Load configuration from file if specified

    Args:
        config_file_path: Path to configuration file

    Returns:
        Dict[str, Any]: Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
    """
    logger = logging.getLogger(__name__)

    try:
        # For simplicity, we'll assume a simple key=value format
        config = {}
        with open(config_file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    logger.warning("Skipping invalid line %d in config file: %s", line_num, line)
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'")  # Remove quotes if present

                config[key] = value
                logger.debug("Loaded config: %s = %s", key, value)

        logger.info(
            "Successfully loaded %d configuration parameters from %s", len(config), config_file_path
        )
        return config

    except FileNotFoundError as e:
        logger.error("Configuration file not found: %s", config_file_path)
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}") from e
    except Exception as e:
        logger.error("Error reading configuration file %s: %s", config_file_path, str(e))
        raise ValueError(f"Invalid configuration file format: {config_file_path}") from e


def create_application_config(args: argparse.Namespace) -> ConfigManager:
    """
    Create and configure the application configuration manager.

    STEP_5: Create ConfigManager instance with application settings

    Args:
        args: Parsed command line arguments

    Returns:
        ConfigManager: Configured configuration manager instance
    """
    logger = logging.getLogger(__name__)

    # Start with file configuration if specified
    base_config = {}
    if args.config_file:
        try:
            base_config = load_config_from_file(args.config_file)
            logger.info("Loaded base configuration from file: %s", args.config_file)
        except (FileNotFoundError, ValueError) as e:
            logger.error("Failed to load config file, using defaults: %s", str(e))

    # Override with command line arguments (these take precedence)
    cmd_config = {"name": args.name, "app_name": APP_NAME, "verbose": args.verbose}
    base_config.update(cmd_config)

    # Create ConfigManager instance
    config_manager = ConfigManager(cfg_dict=base_config)

    # Log final configuration
    logger.info("Application configured with %d parameters", len(config_manager.get_keys()))
    if args.verbose:
        for key in sorted(config_manager.get_keys()):
            value = config_manager.get_value(key)
            logger.info("Config: %s = %s", key, value)

    return config_manager


def main() -> None:
    """
    Main application entry point.

    STEP_6: Main application logic using ConfigManager
    """
    # Parse command line arguments
    args = parse_arguments()

    # Set up logging
    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.info("Starting %s application", APP_NAME)

    try:
        # Create and configure the application
        config = create_application_config(args)

        # Get the name from configuration
        name = config.get_value("name", DEFAULT_NAME)

        # Print the greeting
        greeting = f"Hello {name}"
        print(greeting)

        logger.info("Greeting displayed: %s", greeting)
        logger.info("Application completed successfully")

    except Exception as e:
        logger.exception("Application error occurred: %s", str(e))
        print(f"Error: {str(e)}", file=__import__("sys").stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
