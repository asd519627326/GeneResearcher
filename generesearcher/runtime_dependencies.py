"""Runtime dependency and SciSpaCy model preflight checks."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from importlib import import_module
from typing import Any


REQUIRED_IMPORTS = (
    ("Bio", "biopython"),
    ("click", "click"),
    ("dotenv", "python-dotenv"),
    ("numpy", "numpy"),
    ("openai", "openai"),
    ("pandas", "pandas"),
    ("pymilvus", "pymilvus[milvus_lite]"),
    ("milvus_lite", "milvus-lite"),
    ("requests", "requests"),
    ("scispacy", "scispacy"),
    ("spacy", "spacy"),
    ("tqdm", "tqdm"),
    ("yaml", "PyYAML"),
)

DEFAULT_SCISPACY_MODEL = "en_core_sci_lg"
SCISPACY_MODEL_VERSIONS = {
    "en_core_sci_sm": "0.5.4",
    "en_core_sci_md": "0.5.4",
    "en_core_sci_lg": "0.5.4",
    "en_core_sci_scibert": "0.5.4",
}

IMPORT_DISTRIBUTION_HINTS = {"pkg_resources": "setuptools<81"}


@contextmanager
def _suppress_milvus_lite_pkg_resources_warning():
    """Hide Milvus Lite's known pkg_resources deprecation during its startup."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"pkg_resources is deprecated as an API\.",
            category=UserWarning,
            module=r"milvus_lite",
        )
        yield


def _load_spacy_model(spacy: Any, model_name: str) -> Any:
    """Load a model while hiding one known tokenizer-regex FutureWarning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Possible set union at position \d+",
            category=FutureWarning,
            module=r"spacy\.language",
        )
        return spacy.load(model_name)


def validate_scispacy_model_name(model_name: str) -> str:
    """Return a supported model name or explain the available choices."""
    if not isinstance(model_name, str) or model_name not in SCISPACY_MODEL_VERSIONS:
        choices = ", ".join(SCISPACY_MODEL_VERSIONS)
        raise ValueError(
            f"Unsupported SciSpaCy model '{model_name}'. Choose one of: {choices}."
        )
    return model_name


def validate_runtime_dependencies() -> None:
    """Import required packages so missing or broken installs fail clearly."""
    for module_name, distribution_name in REQUIRED_IMPORTS:
        try:
            if module_name in {"pymilvus", "milvus_lite"}:
                # Milvus Lite can be imported lazily by PyMilvus, and can also
                # emit this warning later when the local client starts.
                with _suppress_milvus_lite_pkg_resources_warning():
                    import_module(module_name)
            else:
                import_module(module_name)
        except Exception as exc:
            if "numpy.dtype size changed" in str(exc):
                raise RuntimeError(
                    "GeneResearcher found a NumPy binary-compatibility error while "
                    f"loading '{module_name}'. The loaded NumPy version does not "
                    "match the compiled spaCy/thinc extensions. The current "
                    "SciSpaCy stack requires NumPy <2; install GeneResearcher in "
                    "a clean Python 3.10–3.12 environment and restart any Jupyter "
                    "kernel after changing compiled dependencies."
                ) from exc
            missing_module = getattr(exc, "name", None)
            if missing_module:
                detail = (
                    f"missing module '{missing_module}' while importing "
                    f"'{module_name}'"
                )
                missing_distribution = IMPORT_DISTRIBUTION_HINTS.get(
                    missing_module,
                    dict(REQUIRED_IMPORTS).get(missing_module, missing_module),
                )
                dependency = f"'{missing_distribution}' ({detail})"
            else:
                dependency = f"'{distribution_name}' ({exc})"
            raise RuntimeError(
                "GeneResearcher cannot start because a required dependency "
                f"could not be imported: {dependency}. Reinstall the package "
                "and its declared dependencies in this environment using the "
                "same install route (for example, reinstall the supplied wheel "
                "or run `python -m pip install -e .` from the source directory)."
            ) from exc


def load_scispacy_model(model_name: str = DEFAULT_SCISPACY_MODEL) -> Any:
    """Load the selected model or raise an actionable setup error."""
    validate_scispacy_model_name(model_name)
    validate_runtime_dependencies()
    spacy = import_module("spacy")
    try:
        return _load_spacy_model(spacy, model_name)
    except OSError as exc:
        raise RuntimeError(
            f"The required SciSpaCy model '{model_name}' is missing "
            "or incompatible with the installed spaCy version. Install or "
            f"repair it with `python -m generesearcher install-spacy-model --model {model_name}`."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"The SciSpaCy model '{model_name}' could not be loaded. "
            f"Install or repair it with `python -m generesearcher install-spacy-model --model {model_name}`, "
            "then check that the installed spaCy version is compatible."
        ) from exc
