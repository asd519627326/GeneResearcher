"""Stable Python API for running one GeneResearcher analysis."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .runtime_dependencies import DEFAULT_SCISPACY_MODEL, validate_scispacy_model_name
from .logging_utils import log_operation
from .core.analysis_lock import AnalysisDirectoryLock

if TYPE_CHECKING:
    from .core.data_models import AnalysisData
else:
    # Keep importing the public API possible before runtime packages are checked.
    AnalysisData = Any


@dataclass
class ResearchConfig:
    """Inputs for one function analysis within a research project.

    Each config runs one function for one or more genes. If ``project_folder``
    is omitted, the run uses temporary storage; a supplied folder persists the
    run and can group multiple functions for the same research background.
    """

    genes: List[str]
    function_name: str
    research_background_template: str
    project_folder: str | Path | None = None
    keywords: List[str] = field(default_factory=list)
    function_query: Optional[str] = None
    exclude_pmids: List[str] = field(default_factory=list)
    publication_date_before: Optional[str] = None
    gene_aliases: Dict[str, List[str]] = field(default_factory=dict)
    max_papers_per_gene: int = 10
    scispacy_model: str = DEFAULT_SCISPACY_MODEL

    def __post_init__(self) -> None:
        if isinstance(self.genes, str) or not self.genes:
            raise ValueError("genes must be a non-empty list of gene names")
        if any(not isinstance(gene, str) or not gene.strip() for gene in self.genes):
            raise ValueError("genes must contain only non-empty strings")
        if not isinstance(self.function_name, str) or not self.function_name.strip():
            raise ValueError("function_name must be a non-empty string")
        if not isinstance(self.research_background_template, str) or not self.research_background_template.strip():
            raise ValueError("research_background_template must be a non-empty string")
        if self.project_folder is not None:
            if not isinstance(self.project_folder, (str, Path)):
                raise TypeError("project_folder must be a string, pathlib.Path, or None")
            if not str(self.project_folder).strip():
                raise ValueError("project_folder must be a non-empty path when provided")
        if not isinstance(self.max_papers_per_gene, int) or isinstance(self.max_papers_per_gene, bool):
            raise TypeError("max_papers_per_gene must be an integer")
        if self.max_papers_per_gene < 1:
            raise ValueError("max_papers_per_gene must be at least 1")
        if not isinstance(self.scispacy_model, str):
            raise TypeError("scispacy_model must be a string")
        validate_scispacy_model_name(self.scispacy_model)
        if isinstance(self.keywords, str) or not isinstance(self.keywords, list):
            raise TypeError("keywords must be a list of search terms")
        if any(not isinstance(keyword, str) or not keyword.strip() for keyword in self.keywords):
            raise ValueError("keywords must contain only non-empty strings")
        if isinstance(self.exclude_pmids, str) or not isinstance(self.exclude_pmids, list):
            raise TypeError("exclude_pmids must be a list of PMID values")
        if any(not isinstance(pmid, str) or not pmid.strip() for pmid in self.exclude_pmids):
            raise ValueError("exclude_pmids must contain only non-empty strings")
        if not isinstance(self.gene_aliases, dict):
            raise TypeError("gene_aliases must map gene symbols to lists of aliases")
        if any(
            not isinstance(gene, str)
            or isinstance(aliases, str)
            or not isinstance(aliases, list)
            or any(not isinstance(alias, str) for alias in aliases)
            for gene, aliases in self.gene_aliases.items()
        ):
            raise ValueError("gene_aliases must map gene symbols to lists of alias strings")
        if isinstance(self.publication_date_before, datetime):
            self.publication_date_before = self.publication_date_before.date().isoformat()
        elif isinstance(self.publication_date_before, date):
            self.publication_date_before = self.publication_date_before.isoformat()
        elif self.publication_date_before is not None and not isinstance(self.publication_date_before, str):
            raise TypeError("publication_date_before must be a date string or date value")

    def to_analysis_data(self) -> AnalysisData:
        """Map the public config onto the canonical pipeline state model."""
        from .core.data_models import AnalysisData

        return AnalysisData(
            genes=list(self.genes),
            function_name=self.function_name,
            research_background_template=self.research_background_template,
            keywords=list(self.keywords),
            function_query=self.function_query,
            exclude_pmids=list(self.exclude_pmids),
            publication_date_before=self.publication_date_before,
            gene_aliases={gene: list(aliases) for gene, aliases in self.gene_aliases.items()},
            max_papers_per_gene=self.max_papers_per_gene,
            scispacy_model=self.scispacy_model,
            project_folder=(
                str(Path(self.project_folder).expanduser())
                if self.project_folder is not None
                else None
            ),
        )


@dataclass
class AnalysisResult:
    """In-memory result, with an owned temporary workspace when applicable."""

    analysis_directory: Path
    report_path: Path
    analysis_data: AnalysisData
    _temporary_workspace: Any = field(default=None, repr=False, compare=False)

    def cleanup(self) -> None:
        """Remove this result's temporary working files, if it owns any."""
        workspace, self._temporary_workspace = self._temporary_workspace, None
        if workspace is not None:
            workspace.cleanup()


class AnalysisSession:
    """Run one function analysis as ordered, individually callable stages.

    A session owns one fresh analysis workspace. Results are persisted to its
    Milvus database after each stage, while ``analysis_data`` exposes the
    current in-memory view for inspection. Stages must run in order. If a stage
    fails, start a new session to retry; a new session clears this function's
    previous database and generated exports. Only one session may use this
    function's analysis directory at a time.
    """

    _STAGE_NAMES = (
        "literature retrieval",
        "sentence-level semantic retrieval",
        "candidate-evidence assessment",
        "evidence synthesis",
        "gene and multi-gene analysis",
        "Markdown report generation",
    )

    def __init__(
        self,
        config: ResearchConfig,
        *,
        env_file: str | Path | None = None,
    ) -> None:
        if not isinstance(config, ResearchConfig):
            raise TypeError("config must be a ResearchConfig instance")

        # Match CLI behavior for library users and keep per-request httpx
        # transport messages out of application log files.
        logging.getLogger("httpx").setLevel(logging.WARNING)

        self.config = config
        self._workspace = None
        if config.project_folder is None:
            self._workspace = tempfile.TemporaryDirectory(prefix="generesearcher-")
            project_folder = Path(self._workspace.name)
        else:
            project_folder = Path(config.project_folder).expanduser().resolve()

        self.analysis_data = config.to_analysis_data()
        self.analysis_data.project_folder = str(project_folder)
        from .core.naming import analysis_directory_name

        self.analysis_directory = project_folder / (
            f"analysis_{analysis_directory_name(config.function_name)}"
        )
        self.report_path = self.analysis_directory / "analysis_report.md"
        self._completed_step = 0
        self._failed_step: int | None = None
        self._closed = False
        self._result_created = False
        self._analysis_lock: AnalysisDirectoryLock | None = None
        self._batch_run_id = uuid.uuid4().hex

        try:
            # Serialize runs that target the same database. Acquire before
            # loading large models or clearing any previous analysis data.
            self._analysis_lock = AnalysisDirectoryLock(
                self.analysis_directory / ".generesearcher.lock",
                self.analysis_directory,
            ).acquire()

            # Check and load SciSpaCy before a session can clear an existing run.
            from .runtime_dependencies import load_scispacy_model

            spacy_nlp = load_scispacy_model(config.scispacy_model)
            from .run import _PipelineExecutor

            self._pipeline = _PipelineExecutor(
                project_folder=str(project_folder),
                env_file=env_file,
                spacy_nlp=spacy_nlp,
            )
        except BaseException:
            self.close()
            raise

    @property
    def completed_steps(self) -> tuple[str, ...]:
        """Names of stages completed successfully in this session."""
        return self._STAGE_NAMES[: self._completed_step]

    @property
    def next_step(self) -> str | None:
        """The next stage name, or ``None`` when report generation is done."""
        if self._completed_step == len(self._STAGE_NAMES):
            return None
        return self._STAGE_NAMES[self._completed_step]

    async def _execute_step(self, step_number: int) -> AnalysisData:
        if not 1 <= step_number <= len(self._STAGE_NAMES):
            raise ValueError(f"Unsupported analysis step: {step_number}.")
        if self._closed:
            raise RuntimeError("This analysis session is closed.")
        if self._result_created:
            raise RuntimeError("This session's result has already been transferred.")
        if self._failed_step is not None:
            failed_name = self._STAGE_NAMES[self._failed_step - 1]
            raise RuntimeError(
                f"The {failed_name} stage failed. Start a new AnalysisSession to retry safely."
            )
        if step_number != self._completed_step + 1:
            expected = self.next_step
            raise RuntimeError(
                f"Stages must run in order; the next stage is {expected!r}, "
                f"not {self._STAGE_NAMES[step_number - 1]!r}."
            )

        try:
            if step_number == 1:
                self.analysis_directory = self._pipeline.prepare_fresh_analysis(
                    self.analysis_data
                )
                self.report_path = self.analysis_directory / "analysis_report.md"
                result = await self._pipeline.run_step_1_retrieval(
                    self.analysis_data,
                    prepared_analysis_directory=self.analysis_directory,
                )
            elif step_number == 2:
                result = await self._pipeline.run_step_2_semantic_filter(
                    self.analysis_data,
                    minimum_cosine_similarity=0.4,
                )
            elif step_number == 3:
                result = await self._pipeline.run_step_3_relevance_filter(
                    self.analysis_data,
                    minimum_cosine_similarity=0.5,
                    batch_run_id=self._batch_run_id,
                )
            elif step_number == 4:
                result = await self._pipeline.run_step_4_synthesis(
                    self.analysis_data,
                    batch_run_id=self._batch_run_id,
                )
            elif step_number == 5:
                result = await self._pipeline.run_step_5_reporting(self.analysis_data)
            else:
                result = await self._pipeline.run_step_6_report_generation(
                    self.analysis_data
                )
        except BaseException:
            self._failed_step = step_number
            self._release_analysis_lock()
            raise

        self.analysis_data = result
        self._completed_step = step_number
        if self._completed_step == len(self._STAGE_NAMES):
            self._release_analysis_lock()
        return result

    async def search_literature(self) -> AnalysisData:
        """Run Step 1: retrieve literature and article text."""
        return await self._execute_step(1)

    async def retrieve_candidate_sentences(self) -> AnalysisData:
        """Run Step 2: embed article text and retrieve candidate sentences."""
        return await self._execute_step(2)

    async def assess_candidate_evidence(self) -> AnalysisData:
        """Run Step 3: assess candidate evidence for relevance."""
        return await self._execute_step(3)

    async def synthesize_evidence(self) -> AnalysisData:
        """Run Step 4: synthesize evidence at article and study levels."""
        return await self._execute_step(4)

    async def analyze_functions(self) -> AnalysisData:
        """Run Step 5: create gene-level and multi-gene reporting results."""
        return await self._execute_step(5)

    async def generate_report(self) -> AnalysisData:
        """Run Step 6: write the final Markdown report."""
        return await self._execute_step(6)

    async def run_all(self) -> AnalysisResult:
        """Continue from the next stage and return the completed analysis."""
        stages = (
            self.search_literature,
            self.retrieve_candidate_sentences,
            self.assess_candidate_evidence,
            self.synthesize_evidence,
            self.analyze_functions,
            self.generate_report,
        )
        while self._completed_step < len(stages):
            await stages[self._completed_step]()
        return self.to_result()

    def to_result(self) -> AnalysisResult:
        """Transfer a completed session into the standard result object."""
        if self._closed:
            raise RuntimeError("This analysis session is closed.")
        if self._completed_step != len(self._STAGE_NAMES):
            raise RuntimeError(
                "The analysis is not complete; finish all six stages before creating a result."
            )
        if self._result_created:
            raise RuntimeError("This session's result has already been transferred.")

        workspace, self._workspace = self._workspace, None
        self._result_created = True
        self._release_analysis_lock()
        return AnalysisResult(
            analysis_directory=self.analysis_directory,
            report_path=self.report_path,
            analysis_data=self.analysis_data,
            _temporary_workspace=workspace,
        )

    def close(self) -> None:
        """Release this session's directory lock and any owned temporary files."""
        if self._closed:
            return
        self._closed = True
        self._release_analysis_lock()
        workspace, self._workspace = self._workspace, None
        if workspace is not None:
            workspace.cleanup()

    def _release_analysis_lock(self) -> None:
        lock, self._analysis_lock = self._analysis_lock, None
        if lock is not None:
            lock.release()

    def __enter__(self) -> AnalysisSession:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


@log_operation("GeneResearcher analysis", log_failure=True)
async def run_analysis_async(
    config: ResearchConfig,
    *,
    env_file: str | Path | None = None,
) -> AnalysisResult:
    """Run all stages in order, using temporary storage unless configured."""
    session = AnalysisSession(config, env_file=env_file)
    try:
        return await session.run_all()
    except BaseException:
        session.close()
        raise


def run_analysis(
    config: ResearchConfig,
    *,
    env_file: str | Path | None = None,
) -> AnalysisResult:
    """Run the pipeline synchronously, using temporary storage by default."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_analysis_async(config, env_file=env_file))
    raise RuntimeError(
        "run_analysis() cannot run inside an active event loop; "
        "use `await run_analysis_async(...)` instead."
    )
