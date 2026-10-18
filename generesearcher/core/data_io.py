# src/core/data_io.py

import json
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from .data_models import AnalysisData
from ..runtime_dependencies import DEFAULT_SCISPACY_MODEL

logger = logging.getLogger(__name__)

def restore_analysis_state(analysis_directory: Path) -> Optional[AnalysisData]:
    """Load pipeline state from Milvus; CSV files are display exports only."""
    logger.debug("Loading analysis parameters from '%s'.", analysis_directory)
    params_path = analysis_directory / "params.json"
    if not params_path.exists():
        logger.warning("Analysis parameters were not found at '%s'.", params_path)
        return None

    try:
        params = json.loads(params_path.read_text(encoding="utf-8"))
        required_fields = {"genes", "function_name", "research_background_template"}
        missing_fields = sorted(required_fields.difference(params))
        if missing_fields:
            raise ValueError(
                "params.json is missing required canonical fields: "
                + ", ".join(missing_fields)
            )
        analysis_data = AnalysisData(
            genes=params.get("genes", []),
            function_name=params.get("function_name", ""),
            research_background_template=params.get("research_background_template", ""),
            keywords=params.get("keywords", []),
            function_query=params.get("function_query"),
            exclude_pmids=params.get("exclude_pmids", []),
            publication_date_before=params.get("publication_date_before"),
            gene_aliases=params.get("gene_aliases", {}),
            scispacy_model=params.get("scispacy_model", DEFAULT_SCISPACY_MODEL),
            max_papers_per_gene=params.get("max_papers_per_gene", 10),
            project_folder=str(analysis_directory.parent),
        )
    except Exception as exc:
        logger.warning("Could not load analysis parameters (%s).", type(exc).__name__)
        return None

    db_path = analysis_directory / "step2_embeddings.db"
    try:
        from ..connectors.milvus_connector import MilvusManager

        # Existing vector collection schemas carry their own dimensions. Recovery
        # does not need to initialize an embedding model or any service key.
        manager = MilvusManager(
            db_path=str(db_path),
            collection_name="biomed_sentences",
            dimension=None,
        )
        try:
            retrieval_status = manager.load_retrieval_status()
            analysis_data.retrieval_status = retrieval_status
            if retrieval_status in {"running", "no_results"}:
                logger.warning(
                    "Step 1 retrieval is %s; ignoring older downstream database rows.",
                    retrieval_status,
                )
                analysis_data.raw_papers_df = pd.DataFrame()
                analysis_data.recalled_sentences_df = pd.DataFrame()
                analysis_data.relevance_filtered_df = pd.DataFrame()
                analysis_data.synthesis_df = pd.DataFrame()
                analysis_data.reporting_data = pd.DataFrame()
                analysis_data.relevance_filter_status = None
            else:
                analysis_data.raw_papers_df = pd.DataFrame(manager.load_retrieved_papers())
                analysis_data.recalled_sentences_df = pd.DataFrame(manager.load_recall_results())

                llm_rows = manager.load_llm_analysis_results()
                for row in llm_rows:
                    result = str(row.get("result", "")).strip().casefold()
                    row["status"] = "success" if result in {"yes", "no"} else (
                        "parse_failed" if result == "parse_failed" else "failed"
                    )
                analysis_data.relevance_filtered_df = pd.DataFrame(llm_rows)
                persisted_step3_status = manager.load_pipeline_stage_status("step3")
                if persisted_step3_status:
                    analysis_data.relevance_filter_status = persisted_step3_status.get("status")
                elif llm_rows:
                    success_count = sum(row["status"] == "success" for row in llm_rows)
                    analysis_data.relevance_filter_status = (
                        "completed" if success_count == len(llm_rows)
                        else "partial" if success_count
                        else "failed"
                    )

                analysis_data.synthesis_df = pd.DataFrame(manager.load_synthesis_results())
                analysis_data.reporting_data = pd.DataFrame(
                    manager.load_reporting_results(analysis_data.function_name)
                )
                if retrieval_status is None:
                    retrieval_status = "completed" if not analysis_data.raw_papers_df.empty else "no_results"
                    manager.save_retrieval_status(retrieval_status)
                    analysis_data.retrieval_status = retrieval_status

        finally:
            manager.close()
    except Exception as exc:
        logger.warning(
            "Could not restore analysis state from Milvus for '%s' (%s).",
            analysis_directory,
            type(exc).__name__,
        )
        return None

    logger.debug(
        "Restored from Milvus: %s papers, %s recall rows, %s Step 3 rows, %s synthesis rows, %s report rows.",
        len(analysis_data.raw_papers_df),
        len(analysis_data.recalled_sentences_df),
        len(analysis_data.relevance_filtered_df),
        len(analysis_data.synthesis_df),
        len(analysis_data.reporting_data),
    )
    return analysis_data



# --- EXPORTING RESULTS ---

class AnalysisSaver:
    """
    A utility class to save all intermediate analysis results in their corresponding directory.
    """
    def __init__(self, analysis_directory: Path):
        self.analysis_directory = analysis_directory
        self.analysis_directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Initialized analysis export writer for '%s'.", self.analysis_directory)

    def save_recalled_sentences(
        self,
        recalled_df: pd.DataFrame,
        similarity_threshold: Optional[float] = None,
    ):
        """
        Saves the DataFrame of recalled sentences to a CSV file.
        The 'text' column is intentionally omitted from the CSV, as the text can be
        retrieved from the Milvus database using the 'sentence_id'.
        """
        if recalled_df.empty:
            logger.warning("The DataFrame of recalled sentences is empty. No file will be saved.")
            return

        # Create a copy to ensure the original DataFrame in memory is not modified.
        df_to_save = recalled_df.copy()

        # --- KEY CHANGE: Drop the 'text' column before saving ---
        if 'text' in df_to_save.columns:
            df_to_save = df_to_save.drop(columns=['text'])
            logger.debug("Temporarily removing 'text' column before saving to CSV.")
        # --- END OF KEY CHANGE ---

        if similarity_threshold is None or similarity_threshold <= 0:
            filename = "step2_embeddings_score_all.csv"
            msg = "all recalled sentences (top_k)"
        else:
            filename = f"step2_embeddings_score_{similarity_threshold}.csv"
            msg = f"filtered recalled sentences (threshold={similarity_threshold})"
        
        file_path = self.analysis_directory / filename
        
        try:
            # Define the desired column order for the CSV file. 'text' is now excluded.
            cols_order = ["Gene", "PMID", "paragraph_id", "sentence_id", "cosine_similarity","text"]
            
            # Ensure a consistent column order.
            # Only include columns that actually exist in the DataFrame.
            final_cols = [col for col in cols_order if col in df_to_save.columns]
            df_to_save = df_to_save.reindex(columns=final_cols)
            
            df_to_save.to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.debug("Exported %s %s record(s) to '%s'.", len(df_to_save), msg, file_path)
        except Exception:
            logger.exception("Could not export recalled sentence records to '%s'.", file_path)





    def save_relevance_results(self, relevance_df: pd.DataFrame):
        """
        Saves the DataFrame of relevance filter results (Step 3) to a CSV file.
        """
        if relevance_df.empty:
            logger.warning("The relevance results DataFrame is empty. No file will be saved.")
            return
            
        filename = "step3_relevance_filtered.csv"
        file_path = self.analysis_directory / filename
        
        try:
            # Sort the display export for easier inspection.
            sorted_df = relevance_df.sort_values(by=["gene", "pmid", "analysis_type"]).reset_index(drop=True)
            sorted_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.debug("Exported %s Step 3 record(s) to '%s'.", len(sorted_df), file_path)
        except Exception:
            logger.exception("Could not export Step 3 relevance results to '%s'.", file_path)




    def save_synthesis_results(self, synthesis_df: pd.DataFrame):
        """
        Saves the DataFrame of synthesis results (Step 4) to a CSV file.
        This function now expects a pre-formatted and pre-sorted DataFrame.
        """
        if synthesis_df.empty:
            logger.warning("The synthesis results DataFrame is empty. No file will be saved.")
            return
            
        filename = "step4_synthesis_results.csv"
        file_path = self.analysis_directory / filename
        
        try:
            # Sorting and formatting are completed in step_04_synthesis.py.
            synthesis_df.to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.debug("Exported %s Step 4 synthesis record(s) to '%s'.", len(synthesis_df), file_path)
        except Exception:
            logger.exception("Could not export Step 4 synthesis results to '%s'.", file_path)

    def save_reporting_results(self, reporting_df: pd.DataFrame):
        """
        Saves the DataFrame of reporting results (Step 5) to a CSV file.
        """
        if reporting_df.empty:
            logger.warning("The reporting results DataFrame is empty. No file will be saved.")
            return

        filename = "step5_reporting_results.csv"
        file_path = self.analysis_directory / filename

        try:
            df_to_save = reporting_df.copy()
            drop_cols = [col for col in ["cot_trace", "model_name"] if col in df_to_save.columns]
            if drop_cols:
                df_to_save = df_to_save.drop(columns=drop_cols)
            df_to_save.to_csv(file_path, index=False, encoding='utf-8-sig')
            logger.debug("Exported %s Step 5 reporting record(s) to '%s'.", len(df_to_save), file_path)
        except Exception:
            logger.exception("Could not export Step 5 reporting results to '%s'.", file_path)
