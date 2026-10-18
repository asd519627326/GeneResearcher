# src/pipeline/step_01_retrieval.py (完整修正版)

import json
import asyncio
import logging
import os
import re
import ast
from pathlib import Path
from datetime import date, datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm.auto import tqdm

from ..connectors.ncbi_connector import NcbiApiConnector, PaperFetchResult
from ..connectors.milvus_connector import MilvusManager
from ..core.data_models import Paper, AnalysisData
from ..core.naming import analysis_directory_name
from ..logging_utils import progress_enabled

logger = logging.getLogger(__name__)


# --- 内部辅助函数 ---

# --- 1. 清洗与格式化函数 ---

def _safe_fix_encoding(text: Any) -> Any:
    """仅在发现常见乱码标记时尝试修复（避免破坏已正确的 Unicode）"""
    if not isinstance(text, str):
        return text
    markers = ['â', 'Ã', 'Â', '�']
    if any(m in text for m in markers):
        try:
            return text.encode('latin1').decode('utf-8')
        except Exception:
            try:
                return text.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')
            except Exception:
                return text
    return text

def _clean_title(title: Any) -> str:
    """清理标题，保留显示友好且兼容 CSV 的形式"""
    if not isinstance(title, str):
        return title if pd.notna(title) else ""
    title = re.sub(r'[\r\n\t]+', ' ', title)
    title = re.sub(r'["""]+', "'", title)
    return title.strip()


def _text_or_empty(value: Any) -> str:
    """Return clean text while treating pandas/NumPy missing scalars as empty."""
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
    return str(value).strip()


def _normalize_article_identifier(value: Any) -> str:
    """Normalize PMID/PMC identifiers for retrieval-completeness checks."""
    normalized = str(value).strip()
    if normalized.upper().startswith("PMID:"):
        normalized = normalized[5:].strip()
    if normalized.upper().startswith("PMC"):
        normalized = "PMC" + normalized[3:]
    return normalized


def _normalize_pmc_source_id(value: Any) -> str:
    normalized = str(value).strip().upper()
    return normalized[3:] if normalized.startswith("PMC") else normalized


def _normalize_list_field(field_value: Any, list_separator: str) -> str:
    """将多种格式的列表（字符串、list等）规范化为用分隔符连接的字符串"""
    
    def clean_text(text):
        if not isinstance(text, str):
            text = str(text)
        return re.sub(r'[\r\n\u2028\u2029]+', ' ', text).strip()
    
    # 1. 首先处理最常见、最正常的输入类型：列表/元组/集合。
    #    这避免了将列表错误地传递给 pd.isna()。
    if isinstance(field_value, (list, tuple, set)):
        return list_separator.join(
            clean_text(item) for item in field_value if str(item).strip()
        )

    # 2. 在确认 x 不是列表后，再安全地检查它是否为标量NA值（None, np.nan等）。
    if pd.isna(field_value):
        return ""

    # 3. 处理字符串输入，尝试解析它。
    if isinstance(field_value, str):
        normalized_value = clean_text(field_value)
        if normalized_value.startswith('[') and normalized_value.endswith(']'):
            try:
                parsed = ast.literal_eval(normalized_value)
                if isinstance(parsed, (list, tuple, set)):
                    return list_separator.join(
                        clean_text(item) for item in parsed if str(item).strip()
                    )
            except Exception: pass
        
        for separator in [';', '|', ', ']:
            if separator in normalized_value:
                parts = [
                    clean_text(part)
                    for part in normalized_value.split(separator)
                    if part.strip()
                ]
                return list_separator.join(parts)
        
        if (
            (normalized_value.startswith('"') and normalized_value.endswith('"'))
            or (normalized_value.startswith("'") and normalized_value.endswith("'"))
        ):
            normalized_value = normalized_value[1:-1].strip()
        return clean_text(normalized_value)

    # 4. 对于其他所有类型（如数字等），作为字符串处理。
    return clean_text(field_value)


# --- 2. 检索与项目设置函数 ---

def _parse_publication_cutoff(date_value: Optional[str]) -> Optional[date]:
    """Parse an ISO publication-date cutoff."""
    if not date_value:
        return None
    try:
        return date.fromisoformat(str(date_value))
    except ValueError as exc:
        raise ValueError(
            "publication_date_before must use ISO format YYYY-MM-DD "
            f"(received {date_value!r})."
        ) from exc


def _build_query(
    database: str,
    gene: str,
    function_name: str,
    keywords: List[str],
    publication_date_before: Optional[str] = None,
    gene_aliases: Optional[List[str]] = None,
) -> str:
    """构建用于PubMed/PMC的布尔查询字符串。
    Gene, function and context keyword groups are always joined with AND.
    Multiple context keywords use OR for PubMed and AND for PMC.

    ``publication_date_before`` is inclusive and limits results to papers
    published on or before that date.
    """
    def wrap(term: str) -> str:
        return f"({term.strip()})"
    gene_terms = [gene, *(gene_aliases or [])]
    gene_terms = [term.strip() for term in gene_terms if term and term.strip()]
    if not gene_terms:
        gene_group = ""
    elif len(gene_terms) == 1:
        gene_group = wrap(gene_terms[0])
    else:
        gene_group = "(" + " OR ".join(wrap(term) for term in gene_terms) + ")"

    function_group = wrap(function_name) if function_name and function_name.strip() else ""
    normalized_keywords = [
        keyword.strip() for keyword in (keywords or []) if keyword and keyword.strip()
    ]
    kw_group = ""
    if normalized_keywords:
        if len(normalized_keywords) == 1:
            kw_group = wrap(normalized_keywords[0])
        else:
            joiner = " OR " if database.lower() == "pubmed" else " AND "
            inner = joiner.join(wrap(keyword) for keyword in normalized_keywords)
            kw_group = "(" + inner + ")"

    query_core = " AND ".join(
        group for group in [gene_group, function_group, kw_group] if group
    )

    parts = [query_core] if query_core else []
    cutoff = _parse_publication_cutoff(publication_date_before)
    if cutoff:
        date_range = (
            f'("1000/01/01"[PDAT] : '
            f'"{cutoff.strftime("%Y/%m/%d")}"[PDAT])'
        )
        parts.append(date_range)
    return " AND ".join(parts) if parts else ""


def _setup_project_structure(
    project_folder: Optional[str], function_name: str, parameters: dict
) -> Path:
    """Create one study root with a shared background and per-function folders."""
    if not project_folder:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_folder = f"Generesearcher_Project_{timestamp}"

    project_root = Path(project_folder)
    project_root.mkdir(parents=True, exist_ok=True)
    requested_background = parameters.get("research_background_template", "")
    study_context_path = project_root / "study_context.json"

    if study_context_path.exists():
        try:
            study_context = json.loads(study_context_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read study context file: {study_context_path}") from exc
        stored_background = study_context.get("research_background_template", "")
        if stored_background != requested_background:
            raise ValueError(
                "This project_folder already belongs to a different research background. "
                "Use a separate project_folder for a different study; multiple functions "
                "and gene sets are allowed within one study."
            )
    else:
        # Infer the study background from canonical function parameters when a
        # project has no root-level study marker yet.
        existing_backgrounds = set()
        for previous_params_path in project_root.glob("analysis_*/params.json"):
            try:
                previous_params = json.loads(previous_params_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if "research_background_template" in previous_params:
                existing_backgrounds.add(previous_params.get("research_background_template", ""))

        if len(existing_backgrounds) > 1 or (
            existing_backgrounds and requested_background not in existing_backgrounds
        ):
            raise ValueError(
                "Existing function folders under this project_folder do not share the "
                "requested research background. Use another project_folder or reconcile "
                "the existing project context first."
            )
        study_context = {"research_background_template": requested_background}
        study_context_path.write_text(
            json.dumps(study_context, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    safe_function_name = analysis_directory_name(function_name)
    analysis_dir = project_root / f"analysis_{safe_function_name}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    params_path = analysis_dir / "params.json"
    if params_path.exists():
        try:
            previous_params = json.loads(params_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous_params = {}
        identity_fields = ("function_name", "research_background_template")
        changed_identity = [
            field for field in identity_fields
            if previous_params.get(field) not in (None, "")
            and parameters.get(field) != previous_params.get(field)
        ]
        if changed_identity:
            raise ValueError(
                "This function directory already belongs to a different function or "
                "research background. Use the matching function directory for this "
                "study; changing the gene set is allowed."
            )
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(parameters, f, indent=4)
    logger.debug("Analysis directory is ready at '%s'.", analysis_dir)
    return analysis_dir

async def _retrieve_and_parse_papers(
    connector: NcbiApiConnector,
    semaphore: asyncio.Semaphore,
    gene: str,
    function_name: str,
    keywords: List[str],
    max_papers_per_gene: int,
    exclude_pmids: List[str],
    publication_date_before: Optional[str],
    gene_aliases: Optional[List[str]],
    show_progress: bool = True,
) -> List[Paper]:
    """为单个基因检索并解析文献。"""
    logger.debug("Searching literature for gene '%s'.", gene)
    pubmed_query = _build_query(
        "pubmed",
        gene,
        function_name,
        keywords,
        publication_date_before,
        gene_aliases,
    )
    pmc_query = _build_query(
        "pmc",
        gene,
        function_name,
        keywords,
        publication_date_before,
        gene_aliases,
    )
    id_tasks = [
        connector.search_ids(pubmed_query, "pubmed", max_papers_per_gene, semaphore),
        connector.search_ids(pmc_query, "pmc", max_papers_per_gene, semaphore)
    ]
    pubmed_ids, pmc_ids = await asyncio.gather(*id_tasks)
    logger.debug(
        "Gene '%s' search returned %s PubMed and %s PMC record ID(s).",
        gene,
        len(pubmed_ids),
        len(pmc_ids),
    )
    configured_pmid_excludes = [
        *[str(value) for value in (exclude_pmids or [])],
        *os.getenv("EXCLUDE_PMIDS", "").split(","),
    ]
    set_pmid_excludes = set()
    for value in configured_pmid_excludes:
        normalized = str(value).strip()
        if normalized.upper().startswith("PMID:"):
            normalized = normalized[5:].strip()
        if normalized.isdecimal():
            set_pmid_excludes.add(normalized)
    total_papers_to_fetch = len(pubmed_ids) + len(pmc_ids)
    if total_papers_to_fetch == 0:
        logger.debug("No records matched the search for gene '%s'.", gene)
        return []
    logger.debug("Fetching details for %s record ID(s) for gene '%s'.", total_papers_to_fetch, gene)
    
    with tqdm(
        total=total_papers_to_fetch,
        desc=f"Fetching for {gene}",
        disable=not show_progress,
    ) as pbar:
        def progress_callback(count: int):
            pbar.update(count)
        detail_tasks = []
        if pubmed_ids:
            detail_tasks.append(("pubmed", pubmed_ids, connector.fetch_papers_by_pubmed_ids(
                pubmed_ids, semaphore, progress_callback
            )))
        if pmc_ids:
            detail_tasks.append(("pmc", pmc_ids, connector.fetch_papers_by_pmc_ids(
                pmc_ids, semaphore, progress_callback
            )))
        results = await asyncio.gather(
            *(task for _, _, task in detail_tasks),
            return_exceptions=True,
        )
        
    all_papers = {}
    pmc_source_id_to_paper_id = {}
    failed_detail_sources = []
    for (source, requested_ids, _), result in zip(detail_tasks, results):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException):
            if not isinstance(result, Exception):
                raise result
            failed_detail_sources.append((source, requested_ids, result))
            logger.warning(
                "Could not fetch %s article details for gene '%s' (%s).",
                source,
                gene,
                type(result).__name__,
            )
            continue
        if not isinstance(result, PaperFetchResult):
            raise TypeError(
                f"The {source} detail fetch returned an unsupported result type: "
                f"{type(result).__name__}."
            )

        if source == "pmc":
            pmc_source_id_to_paper_id.update(result.source_id_to_paper_id)

        for paper in result.papers:
            if not paper.paper_id:
                continue

            paper_id = _normalize_article_identifier(paper.paper_id)
            paper_pmid = paper_id[5:].strip() if paper_id.upper().startswith("PMID:") else paper_id
            if paper_pmid.isdecimal() and paper_pmid in set_pmid_excludes:
                logger.debug("Excluded PMID %s using the configured exclusion list.", paper_pmid)
                continue

            existing_paper = all_papers.get(paper_id)
            if existing_paper is None:
                all_papers[paper_id] = paper
                continue

            # Keep the richest copy when PubMed and PMC return the same article.
            if paper.abstract and not existing_paper.abstract:
                existing_paper.abstract = paper.abstract
            if paper.full_text and not existing_paper.full_text:
                existing_paper.full_text = paper.full_text
                existing_paper.source = paper.source
            if paper.title and (
                not existing_paper.title
                or existing_paper.title == "Title not available from PubMed"
            ):
                existing_paper.title = paper.title

    available_paper_ids = set(all_papers)
    missing_pmids = sorted({
        _normalize_article_identifier(pmid)
        for pmid in pubmed_ids
        if _normalize_article_identifier(pmid) not in available_paper_ids
        and _normalize_article_identifier(pmid) not in set_pmid_excludes
    })
    missing_pmcids = []
    for pmc_id in sorted({_normalize_pmc_source_id(value) for value in pmc_ids}):
        canonical_id = pmc_source_id_to_paper_id.get(pmc_id)
        if canonical_id:
            canonical_id = _normalize_article_identifier(canonical_id)
        else:
            canonical_id = f"PMC{pmc_id}"

        if canonical_id in available_paper_ids:
            continue
        if canonical_id.isdecimal() and canonical_id in set_pmid_excludes:
            continue
        missing_pmcids.append(pmc_id)

    missing_details = []
    if missing_pmids:
        missing_details.append(
            f"{len(missing_pmids)} PMID(s): {', '.join(missing_pmids[:10])}"
        )
    if missing_pmcids:
        missing_details.append(
            f"{len(missing_pmcids)} PMC search ID(s): {', '.join(missing_pmcids[:10])}"
        )
    if missing_details:
        source_failures = ", ".join(
            f"{source} ({type(error).__name__})"
            for source, _, error in failed_detail_sources
        )
        failure_suffix = f" Detail request failures: {source_failures}." if source_failures else ""
        raise RuntimeError(
            f"Step 1 article-detail retrieval was incomplete for gene '{gene}': "
            + "; ".join(missing_details)
            + ". No partial result set was accepted."
            + failure_suffix
        )
            
    # max_papers_per_gene limits each source search independently (PubMed and PMC).
    # After merging, retain every unique article instead of truncating PMC results.
    return list(all_papers.values())


# --- 主函数 (对外暴露的接口) ---

async def run_retrieval_step(
    services: Dict[str, Any],
    analysis_data: AnalysisData,
    prepared_analysis_dir: Optional[Path] = None,
) -> AnalysisData:
    """
    执行数据检索、清洗、格式化和存储的完整第一步流程。
    """
    analysis_data.retrieval_status = "running"
    
    # --- 1. 初始化 ---
    ncbi_connector = services["ncbi_connector"]
    api_semaphore = asyncio.Semaphore(3)
    analysis_dir = prepared_analysis_dir or _setup_project_structure(
        analysis_data.project_folder,
        analysis_data.function_name,
        analysis_data.to_persisted_parameters(services),
    )
    db_path = analysis_dir / "step2_embeddings.db"
    state_manager = MilvusManager(
        db_path=str(db_path),
        collection_name="biomed_sentences",
        dimension=services["embedding_manager"].dimensions,
    )
    try:
        state_manager.save_retrieval_status("running")
        state_manager.save_pipeline_stage_status("step3", "stale")
    finally:
        state_manager.close()

    # --- 2. 并发检索文献 ---
    all_papers_list = []
    total_genes = len(analysis_data.genes)
    for gene_index, gene in enumerate(analysis_data.genes, start=1):
        logger.info(
            "Step 1: retrieving relevant literature for target gene '%s' (%s/%s).",
            gene,
            gene_index,
            total_genes,
        )
        papers_for_gene = await _retrieve_and_parse_papers(
            connector=ncbi_connector,
            semaphore=api_semaphore,
            gene=gene,
            function_name=analysis_data.function_query or analysis_data.function_name,
            keywords=analysis_data.keywords,
            max_papers_per_gene=analysis_data.max_papers_per_gene,
            exclude_pmids=analysis_data.exclude_pmids,
            publication_date_before=analysis_data.publication_date_before,
            gene_aliases=analysis_data.gene_aliases.get(gene, []),
            show_progress=progress_enabled(logger),
        )
        if papers_for_gene:
            for paper in papers_for_gene:
                paper_dict = paper.to_dict()
                paper_dict['Gene'] = gene
                all_papers_list.append(paper_dict)
    
    if not all_papers_list:
        logger.warning(
            "No relevant literature was retrieved for any target gene; Step 1 stopped before sentence-level semantic retrieval."
        )
        analysis_data.raw_papers_df = pd.DataFrame()
        analysis_data.retrieval_status = "no_results"
        state_manager = MilvusManager(
            db_path=str(db_path),
            collection_name="biomed_sentences",
            dimension=services["embedding_manager"].dimensions,
        )
        try:
            state_manager.save_retrieval_status("no_results")
        finally:
            state_manager.close()
        return analysis_data

    # --- 3. 数据清洗与格式化 ---
    logger.debug("Retrieved %s gene-article record(s); preparing canonical database records.", len(all_papers_list))
    # Keep the connector's original values as the canonical database input.
    # Display cleanup below may repair encoding or flatten title whitespace;
    # those transformations must not replace the source strings in Milvus.
    source_df = pd.DataFrame(all_papers_list).copy(deep=True)
    df = source_df.copy(deep=True)
    
    text_cols = ['title', 'abstract', 'full_text', 'keywords']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(_safe_fix_encoding)
            
    LIST_SEPARATOR = "; "
    df['title'] = df['title'].apply(_clean_title)
    logger.debug("Cleaning and formatting complete.")

    # --- 4. 导出摘要/全文为TXT，并生成最终CSV ---
    output_dir_txt = analysis_dir / "references"
    output_dir_txt.mkdir(parents=True, exist_ok=True)
    output_csv_path = analysis_dir / "step1_retrieved_papers_metadata.csv"

    db_source_paths = []
    skipped_count = 0
    for idx, row in df.iterrows():
        pmid = str(row.get('paper_id', '')).strip()
        gene = str(row.get('Gene', '')).strip()

        abstract_clean = _text_or_empty(row.get('abstract'))
        full_text_clean = _text_or_empty(row.get('full_text'))

        has_content = (abstract_clean != "") or (
            full_text_clean != "" and full_text_clean != "No structured body prose found."
        )

        if not pmid or not has_content:
            skipped_count += 1
            db_source_paths.append("")
            continue

        content = (
            f"# Abstract:\n{abstract_clean}\n\n"
            f"# Full Text:\n{full_text_clean}\n\n"
        )

        safe_gene = "".join(c for c in gene if c.isalnum() or c in ('-', '_'))
        filename = f"{safe_gene}_{pmid}.txt"
        filepath = output_dir_txt / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as exc:
            logger.error(
                "Could not export reference text for PMID %s / gene %s to '%s' (%s).",
                pmid,
                gene,
                filepath,
                type(exc).__name__,
            )
            raise

        db_source_paths.append(str(filepath.relative_to(analysis_dir)))

    if skipped_count > 0:
        logger.warning(
            "Skipped %s reference export(s) because the record lacked a PMID or usable article text.",
            skipped_count,
        )

    df['db_source'] = db_source_paths
    df.rename(columns={'paper_id': 'PMID'}, inplace=True)

    # Store the unformatted source records in Milvus. CSV/TXT below are display
    # exports only; pipeline stages and recovery read their canonical values here.
    source_df['db_source'] = db_source_paths
    source_df.rename(columns={'paper_id': 'PMID'}, inplace=True)
    milvus_manager = MilvusManager(
        db_path=str(db_path),
        collection_name="biomed_sentences",
        dimension=services["embedding_manager"].dimensions,
    )
    try:
        milvus_manager.save_retrieved_papers(source_df)
        milvus_manager.save_retrieval_status("completed")
        active_pairs = {
            (str(row.get("PMID", "")), str(row.get("Gene", "")))
            for _, row in source_df.iterrows()
        }
        analysis_data.raw_papers_df = pd.DataFrame([
            row for row in milvus_manager.load_retrieved_papers()
            if (str(row.get("PMID", "")), str(row.get("Gene", ""))) in active_pairs
        ])
        persisted_counts = (
            analysis_data.raw_papers_df.groupby("Gene")["PMID"].nunique().to_dict()
            if not analysis_data.raw_papers_df.empty
            else {}
        )
        gene_breakdown = "; ".join(
            f"{gene}: {int(persisted_counts.get(gene, 0))} article(s)"
            for gene in analysis_data.genes
        )
        unique_pmid_count = (
            analysis_data.raw_papers_df["PMID"].nunique()
            if not analysis_data.raw_papers_df.empty
            else 0
        )
        logger.info(
            "Step 1 literature retrieval persisted %s of %s gene-article record(s) covering %s unique PMID(s): %s.",
            len(analysis_data.raw_papers_df),
            len(source_df),
            unique_pmid_count,
            gene_breakdown,
        )
        if len(analysis_data.raw_papers_df) != len(source_df):
            logger.warning(
                "Milvus returned %s of %s expected Step 1 record(s).",
                len(analysis_data.raw_papers_df),
                len(source_df),
            )
    finally:
        milvus_manager.close()
    
    final_columns = [
       'PMID', 'title', 'authors', 'publication_types', 'doi', 'published_date', 'url',
        'source', 'updated_date', 'keywords', 'Gene', 'db_source'
    ]
    df_display = df.copy()
    for col in ['keywords', 'authors', 'publication_types']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda field_value: _normalize_list_field(field_value, LIST_SEPARATOR)
            )
    df_out = df_display[[col for col in final_columns if col in df_display.columns]]
    
    try:
        df_out.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    except Exception as exc:
        logger.error(
            "Could not export Step 1 article metadata to '%s' (%s).",
            output_csv_path,
            type(exc).__name__,
        )
        raise
    logger.debug("Reference text files were exported to '%s'.", output_dir_txt)
    logger.debug("Article metadata was exported to '%s'.", output_csv_path)

    # --- 5. 更新并返回AnalysisData对象 ---
    analysis_data.retrieval_status = "completed"
    return analysis_data
