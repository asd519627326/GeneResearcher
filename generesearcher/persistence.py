"""Optional save and restore helpers for completed analyses."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from .api import AnalysisResult


def _validate_analysis_directory(analysis_directory: Path) -> None:
    if not analysis_directory.is_dir():
        raise FileNotFoundError(f"Analysis directory does not exist: {analysis_directory}")

    required_files = ("params.json", "step2_embeddings.db")
    missing_files = [name for name in required_files if not (analysis_directory / name).is_file()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Analysis directory is missing required file(s): {missing} ({analysis_directory})"
        )


def _validate_project_background(project_root: Path, research_background: str) -> None:
    """Prevent adding a function analysis to a different study project."""
    context_path = project_root / "study_context.json"
    if context_path.is_file():
        try:
            context = json.loads(context_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read project context file: {context_path}") from exc
        if context.get("research_background_template") != research_background:
            raise ValueError(
                "The destination project folder belongs to a different research background. "
                "Choose another project folder."
            )
        return

    existing_backgrounds = set()
    for params_path in project_root.glob("analysis_*/params.json"):
        try:
            params = json.loads(params_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read saved analysis parameters: {params_path}") from exc
        if "research_background_template" in params:
            existing_backgrounds.add(params["research_background_template"])

    if existing_backgrounds and existing_backgrounds != {research_background}:
        raise ValueError(
            "Existing analyses in the destination folder use a different research "
            "background. Choose another project folder."
        )


def save_analysis(result: AnalysisResult, destination: str | Path) -> Path:
    """Save one function analysis inside a user-selected study project folder.

    The project root can hold multiple ``analysis_<function>`` directories
    that share the same research background. The Milvus Lite database and all
    generated files are copied directly; display CSV files are not used to
    reconstruct the analysis state. The function directory must not already
    exist in the destination project.
    """
    from .api import AnalysisResult

    if not isinstance(result, AnalysisResult):
        raise TypeError("result must be an AnalysisResult returned by run_analysis()")

    source = Path(result.analysis_directory).expanduser().resolve()
    _validate_analysis_directory(source)

    project_root = Path(destination).expanduser().resolve()
    if project_root.exists() and not project_root.is_dir():
        raise NotADirectoryError(f"Project folder is not a directory: {project_root}")
    if project_root == source or source in project_root.parents:
        raise ValueError("The destination project folder must be outside the source analysis directory")
    if source.parent == project_root:
        return source

    research_background = result.analysis_data.research_background_template
    if project_root.exists():
        _validate_project_background(project_root, research_background)
    else:
        project_root.parent.mkdir(parents=True, exist_ok=True)
        project_root.mkdir()

    from .core.naming import analysis_directory_name

    analysis_name = f"analysis_{analysis_directory_name(result.analysis_data.function_name)}"
    target = project_root / analysis_name
    if target.exists():
        raise FileExistsError(
            f"This function already has a saved analysis in '{project_root}': {target}"
        )

    staging = project_root / f".{analysis_name}.saving-{uuid4().hex}"
    context_staging = None
    try:
        shutil.copytree(source, staging, copy_function=shutil.copy2, symlinks=True)
        os.replace(staging, target)
        context_path = project_root / "study_context.json"
        if not context_path.exists():
            context_staging = project_root / f".study_context.saving-{uuid4().hex}"
            context_staging.write_text(
                json.dumps(
                    {"research_background_template": research_background},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            os.replace(context_staging, context_path)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if context_staging is not None and context_staging.exists():
            context_staging.unlink()
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        raise

    result.analysis_directory = target
    result.report_path = target / "analysis_report.md"
    result.analysis_data.project_folder = str(project_root)
    result.cleanup()
    return target


def load_analysis(source: str | Path) -> AnalysisResult:
    """Restore a saved analysis directory into an ``AnalysisResult``.

    Restoration reads parameters and analysis state from ``params.json`` and
    the Milvus Lite database. It does not load CSV exports or call external
    literature, embedding, or language-model services.
    """
    from .api import AnalysisResult
    from .core.data_io import restore_analysis_state

    analysis_directory = Path(source).expanduser().resolve()
    _validate_analysis_directory(analysis_directory)

    analysis_data = restore_analysis_state(analysis_directory)
    if analysis_data is None:
        raise RuntimeError(
            f"Could not restore analysis state from '{analysis_directory}'. "
            "Check params.json and step2_embeddings.db."
        )

    return AnalysisResult(
        analysis_directory=analysis_directory,
        report_path=analysis_directory / "analysis_report.md",
        analysis_data=analysis_data,
    )
