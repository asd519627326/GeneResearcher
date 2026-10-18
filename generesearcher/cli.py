"""Command line interface for GeneResearcher."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

from .api import ResearchConfig, run_analysis_async
from .runtime_dependencies import DEFAULT_SCISPACY_MODEL, SCISPACY_MODEL_VERSIONS

logger = logging.getLogger(__name__)


def _load_research_config(config_path: Path) -> ResearchConfig:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to read study YAML files. Reinstall GeneResearcher "
            "with `python -m pip install generesearcher`."
        ) from exc

    try:
        raw_config: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read configuration file: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration file: {config_path}") from exc

    if not isinstance(raw_config, dict):
        raise ValueError("The YAML configuration must contain a mapping of fields.")

    configured_project_folder = raw_config.get("project_folder")
    if configured_project_folder is not None:
        configured_project_folder = Path(configured_project_folder).expanduser()
        if not configured_project_folder.is_absolute():
            configured_project_folder = config_path.parent / configured_project_folder
        raw_config["project_folder"] = str(configured_project_folder.resolve())
    return ResearchConfig(**raw_config)


def _resolve_env_file(explicit_path: str | None, config_directory: Path) -> Path | None:
    if explicit_path:
        env_path = Path(explicit_path).expanduser()
        if not env_path.is_absolute():
            env_path = config_directory / env_path
        return env_path.resolve()

    config_directory_env = config_directory / ".env"
    if config_directory_env.is_file():
        return config_directory_env
    working_directory_env = Path.cwd() / ".env"
    return working_directory_env if working_directory_env.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="generesearcher")
    subparsers = parser.add_subparsers(dest="command", required=True)
    install_model_parser = subparsers.add_parser(
        "install-spacy-model",
        help="download a SciSpaCy model used by Step 2",
    )
    install_model_parser.add_argument(
        "--model",
        choices=SCISPACY_MODEL_VERSIONS,
        default=DEFAULT_SCISPACY_MODEL,
        help=f"SciSpaCy model to install (default: {DEFAULT_SCISPACY_MODEL})",
    )
    run_parser = subparsers.add_parser("run", help="run one configured function analysis")
    run_parser.add_argument("--config", required=True, type=Path, help="path to a study YAML file")
    run_parser.add_argument("--env-file", help="path to the credentials and service configuration file")
    run_parser.add_argument(
        "--save-to",
        type=Path,
        help="save results under this study project folder; without it, results use temporary storage",
    )
    log_options = run_parser.add_mutually_exclusive_group()
    log_options.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "OFF"),
        dest="log_level",
        help="set the minimum log level (default: INFO; OFF disables logging)",
    )
    log_options.add_argument(
        "--verbose",
        action="store_const",
        const="DEBUG",
        dest="log_level",
        help="show detailed diagnostic logs (same as --log-level DEBUG)",
    )
    log_options.add_argument(
        "--quiet",
        action="store_const",
        const="ERROR",
        dest="log_level",
        help="show errors only",
    )
    run_parser.set_defaults(log_level="INFO")
    arguments = parser.parse_args(argv)

    if arguments.command == "install-spacy-model":
        from .model_installer import install_scispacy_model

        try:
            install_scispacy_model(arguments.model)
        except (RuntimeError, ValueError) as exc:
            parser.exit(1, f"Error: {exc}\n")
        return 0

    configured_log_level = (
        logging.CRITICAL + 1
        if arguments.log_level == "OFF"
        else getattr(logging, arguments.log_level)
    )
    logging.basicConfig(
        level=configured_log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    for handler in logging.getLogger().handlers:
        handler.setLevel(configured_log_level)
    # Keep verbose third-party transport and database chatter out of normal
    # application logs. Users who embed the package can configure these loggers.
    for dependency_logger in ("httpx", "openai", "pymilvus"):
        logging.getLogger(dependency_logger).setLevel(logging.WARNING)

    try:
        config_path = arguments.config.expanduser().resolve()
        research_config = _load_research_config(config_path)
        if arguments.save_to is not None:
            save_path = arguments.save_to.expanduser()
            if not save_path.is_absolute():
                save_path = config_path.parent / save_path
            research_config.project_folder = str(save_path.resolve())
        env_file = _resolve_env_file(arguments.env_file, config_path.parent)
    except Exception as exc:
        logger.error(
            "Could not load analysis configuration '%s' (%s).",
            arguments.config,
            type(exc).__name__,
        )
        return 2

    try:
        result = asyncio.run(run_analysis_async(research_config, env_file=env_file))
    except Exception:
        # The public API logs analysis failures once with the traceback.
        if research_config.project_folder is not None:
            logger.error("Partial analysis data, if any, remain under '%s'.", research_config.project_folder)
        return 1
    if research_config.project_folder is None:
        print(
            "Analysis completed in temporary storage. Results will be removed when this command exits; "
            "use --save-to PATH to keep them."
        )
    else:
        print(f"Analysis directory: {result.analysis_directory}")
        print(f"Report: {result.report_path}")
    result.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
