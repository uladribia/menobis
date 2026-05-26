"""Logging configuration for MENoBiS applications and CLI entry points."""

from pathlib import Path

from loguru import logger

DEFAULT_LOG_FILENAME = "menobis.log"
DEFAULT_ROTATION = "5 MB"
DEFAULT_RETENTION = "30 days"


def configure_logging(
    log_dir: Path | str,
    *,
    console: bool = True,
    level: str = "INFO",
) -> None:
    """Configure Loguru logging for MENoBiS.

    Args:
        log_dir: Directory where the rotating log file is written.
        console: Whether to keep stderr console logging enabled.
        level: Minimum log level for configured handlers.

    """
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / DEFAULT_LOG_FILENAME

    logger.remove()
    if console:
        logger.add(
            sink=lambda message: print(message, end=""),
            level=level,
            enqueue=True,
        )
    logger.add(
        log_path,
        level=level,
        rotation=DEFAULT_ROTATION,
        retention=DEFAULT_RETENTION,
        enqueue=False,
    )


__all__ = [
    "DEFAULT_LOG_FILENAME",
    "DEFAULT_RETENTION",
    "DEFAULT_ROTATION",
    "configure_logging",
]
