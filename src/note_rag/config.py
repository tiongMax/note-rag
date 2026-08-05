"""Shared application environment loading."""

from os import PathLike
from pathlib import Path

from dotenv import load_dotenv


def load_environment(dotenv_path: str | PathLike[str] | None = None) -> bool:
    """Load a project .env without replacing real process environment values."""

    if dotenv_path is not None:
        return load_dotenv(dotenv_path=dotenv_path, override=False)

    candidates = (
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    )
    for candidate in candidates:
        if candidate.is_file():
            return load_dotenv(dotenv_path=candidate, override=False)
    return False
