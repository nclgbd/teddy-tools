"""
General utility functions. These are not specific to any deep learning framework, and therefore can be used in different contexts.
"""

# common imports
import os
import yaml

# logging
import logging
from logging import Logger

# rich
import rich
from rich.console import Console
from rich.logging import RichHandler

__all__ = [
    "_logger",
    "_logger_format",
    "get_console",
    "get_logger",
    "login",
    "repl",
]


_logger_format: logging.Formatter = logging.Formatter(fmt="%(name)s: %(message)s")


def get_logger(name: str = None, level: int = logging.INFO):
    """
    Function to get a logger with a `RichHandler`. Sets up the logger with a custom format and a `StreamHandler`.
    ## Args:
        `name` (`str`): The name of the logger. Defaults to `None`.
        `level` (`int`): The level of the logger. Defaults to `logging.INFO`.

    ## Returns:
        `logging.Logger`: The logger.
    """

    logger: Logger = logging.getLogger(name)
    logger.setLevel(level=level)
    # stream_handler = logging.StreamHandler(stream=sys.stdout)
    rich_handler = RichHandler(rich_tracebacks=True, level=level)

    # stream_handler.setFormatter(_logger_format)
    rich_handler.setFormatter(_logger_format)
    # logger.addHandler(stream_handler)
    logger.addHandler(rich_handler)
    logger.propagate = False

    return logger


def get_console():
    """
    Gets the rich console object.

    ## Returns:
        `Console`: Rich console object.
    """

    _console = Console()
    return _console


def load_yaml(path: os.PathLike = "./repl.yml"):
    """
    Loads a yaml file.

    ## Args:
        `path` (`os.PathLike`): Path to the yaml file. Defaults to `"repl.yml"`.

    ## Returns:
        `dict`: The yaml file as a dictionary.
    """

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


_logger: Logger = get_logger()
