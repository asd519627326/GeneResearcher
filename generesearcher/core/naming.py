"""Shared naming helpers for research and analysis directories."""


def analysis_directory_name(function_name: str) -> str:
    """Return a filesystem-safe suffix for one analyzed function."""
    return function_name.lower().replace(" ", "_").replace("/", "_")
