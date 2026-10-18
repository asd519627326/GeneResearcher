"""Install a SciSpaCy model used by Step 2."""

from __future__ import annotations

import importlib
import subprocess
import sys

from .runtime_dependencies import (
    DEFAULT_SCISPACY_MODEL,
    SCISPACY_MODEL_VERSIONS,
    _load_spacy_model,
    load_scispacy_model,
    validate_runtime_dependencies,
    validate_scispacy_model_name,
)


SCISPACY_MODEL_ARCHIVE_URL = (
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/"
    "v{version}/{model_name}-{version}.tar.gz"
)


def install_scispacy_model(model_name: str = DEFAULT_SCISPACY_MODEL) -> None:
    """Download a pinned SciSpaCy model into the active Python environment."""
    validate_scispacy_model_name(model_name)
    validate_runtime_dependencies()
    spacy = importlib.import_module("spacy")
    model_version = SCISPACY_MODEL_VERSIONS[model_name]
    model_url = SCISPACY_MODEL_ARCHIVE_URL.format(
        version=model_version,
        model_name=model_name,
    )

    try:
        _load_spacy_model(spacy, model_name)
    except Exception:
        model_is_installed = spacy.util.is_package(model_name)
        command = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
        if model_is_installed:
            command.extend(["--force-reinstall", "--no-deps"])
        command.append(model_url)
        print(
            f"Downloading SciSpaCy model {model_name} {model_version}...",
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "SciSpaCy model installation failed. Check the download output "
                f"and retry `python -m generesearcher install-spacy-model --model {model_name}`."
            )
    else:
        print(f"SciSpaCy model already installed: {model_name}")
        return

    # Validate the installed artifact through the same loader used by the main API.
    load_scispacy_model(model_name)
    print(f"SciSpaCy model ready: {model_name}")
