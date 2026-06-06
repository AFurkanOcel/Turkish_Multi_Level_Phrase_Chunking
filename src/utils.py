"""Shared utility functions for the project."""

from pathlib import Path


def ensure_directory(path):
    """Create a directory if it does not already exist and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def project_path(*parts):
    """Build a path relative to the project root."""
    project_root = Path(__file__).resolve().parents[1]
    return project_root.joinpath(*parts)
