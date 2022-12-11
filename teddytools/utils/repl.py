"""
Sets up the REPL for debugging.
"""

# rich
import rich
from rich import pretty
from rich import traceback

# local imports
from teddytools.utils import get_console


def install(
    console: rich.console.Console = None,
    max_depth: int = None,
    max_length: int = None,
    max_string: int = None,
    show_locals: bool = False,
    width: int = 100,
):
    """
    Install automatic formatting for `rich.pretty` and `rich.traceback` modules.

    ## Args:
        `console` (`Console`, optional): Defaults to `None`.
        `max_depth` (`int`, optional): Defaults to `None`.
        `max_length` (`int`, optional): Defaults to `None`.
        `max_string` (`int`, optional): Defaults to `None`.
        `show_locals` (`bool`, optional): Defaults to `False`.
        `width` (`int`, optional): Defaults to `100`.
    """
    console = get_console()

    pretty.install(
        console=console,
        max_depth=max_depth,
        max_length=max_length,
        max_string=max_string,
    )
    traceback.install(
        console=console,
        show_locals=show_locals,
        width=width,
    )

install()
