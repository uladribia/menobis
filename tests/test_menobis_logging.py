"""Tests for MENoBiS logging configuration."""

from pathlib import Path

from loguru import logger

from menobis.utilities.logging import DEFAULT_LOG_FILENAME, configure_logging


def test_configure_logging_creates_rotating_log_file(tmp_path: Path) -> None:
    """Logging configuration writes to the requested directory."""
    configure_logging(log_dir=tmp_path, console=False)

    logger.info("test message")

    log_path = tmp_path / DEFAULT_LOG_FILENAME
    assert log_path.exists()
    assert "test message" in log_path.read_text()
