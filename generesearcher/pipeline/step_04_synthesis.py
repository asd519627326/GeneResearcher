# src/pipeline/step_04_synthesis.py

import asyncio
import hashlib
import json
import logging
import re
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from collections import defaultdict
import os
import time

# --- Relative imports from within the project ---
from ..core.data_models import AnalysisData
from ..connectors.llm_api import (
    BATCH_API_ENABLED,
    BATCH_SUPPORTED_MODELS,
    LLMManager,
    get_llm_response_content,
)
from ..core.data_io import AnalysisSaver
from ..core.naming import analysis_directory_name
from ..connectors.milvus_connector import MilvusManager
from ..connectors.embedding_connector import EmbeddingManager
from ..connectors.batch_manager import BatchManager, BATCH_INACTIVITY_TIMEOUT_SECONDS
from ..logging_utils import progress_enabled
# --- Module-level constants ---
logger = logging.getLogger(__name__)
MAX_CONCURRENT_TASKS = 3
MAX_RETRIES = 5
RETRY_DELAY = 30
RELATIONSHIP_LABELS = {
    "gene_vs_function": "G–F",
    "gene_vs_context": "G–C",
    "function_vs_context": "F–C",
}


class BatchInactivityTimeout(TimeoutError):
    """Signals that Step 4 should stop using Batch and finish through realtime calls."""



# ... (Las funciones auxiliares _parse_condensation_output, _run_single_document_condensation, 
#      _run_cross_document_synthesis permanecen IGUALES. Las omito por brevedad.)
# =================================================================
# === 1. Helper Function for Parsing LLM Output ===
# =================================================================

def _clean_csv_text(text: Any) -> str:
    """
    Cleans text for CSV compatibility by removing excessive whitespace.
    Handles non-string inputs gracefully.
    """
    if not isinstance(text, str):
        return str(text)
    # Reemplaza saltos de línea, tabulaciones y múltiples espacios con un solo espacio
    return re.sub(r'\s+', ' ', text).strip()


def _extract_pmids_from_text(text: str) -> List[str]:
    """Extracts unique PMIDs (e.g., [PMID:12345]) from text."""
    if not text: return []
    matches = re.findall(r'PMID:?\s*(\d+)', text, re.IGNORECASE)
    return sorted(list(set(matches)))



def _extract_and_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extracts a JSON object from a string that might be wrapped in code blocks.
    (Ported from step_03 to keep this stage independent)
    """
    if not text:
        return None
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    logger.warning("Could not parse the Step 4 response as a JSON object.")
    return None


def _parse_condensation_output(text: str) -> tuple[str, List[str]]:
    """Validate the Step 4 response contract before accepting a summary."""
    parsed = _extract_and_parse_json(text)
    if not isinstance(parsed, dict):
        raise ValueError("Response is not a valid JSON object")
    summary = parsed.get("summary")
    evidence = parsed.get("evidence")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("JSON response must contain a non-empty string 'summary'")
    if not isinstance(evidence, (list, str)):
        raise ValueError("JSON response must contain an 'evidence' list or string")
    return summary.strip(), _safe_extract_evidence_list(parsed)



def _safe_extract_evidence_list(parsed_json: Dict[str, Any]) -> List[str]:
    """
    Defensive coding to ensure we get a list of strings for evidence,
    even if the LLM returns a string or single item.
    """
    evidence = parsed_json.get("evidence", [])
    
    # Case A: It is already a list (Expected behavior)
    if isinstance(evidence, list):
        # Ensure all elements are strings and clean them
        return [str(item).strip() for item in evidence if item]

    # Case B: It's a string (LLM hallucinated format, e.g., "1. xxx\n2. xxx")
    if isinstance(evidence, str):
        # Split by newline or semicolon, remove bullets
        cleaned_lines = []
        # Normalize newlines
        raw_lines = evidence.replace(';', '\n').split('\n')
        for line in raw_lines:
            # Remove common bullet points like "1.", "-", "•"
            cleaned = re.sub(r'^[\d\.\-\•\s]+', '', line).strip()
            if cleaned:
                cleaned_lines.append(cleaned)
        return cleaned_lines

    # Case C: Other types (None, int, etc.)
    return []



def _reconstruct_background(
    background_template: str,
    gene: str,
    function_name: str,
    analysis_type: str,
) -> Optional[str]:
    """
    Re-generates the background context string based on the analysis type.
    Logic mirrors '_prepare_analysis_target' from step_03_relevance_filter.py
    """
    if analysis_type == "gene_vs_function":
        # In Step 3, this returns {'background': None}
        return None
        
    if analysis_type == "gene_vs_context":
        # Step 3 logic: template + note, then formatted with gene
        processed_template = background_template + '(Note: Please ignore the reasoning through {function_name})'
        context = processed_template.format(gene=gene, function_name="[MASKED FUNCTION]")
        return context
        
    if analysis_type == "function_vs_context":
        # Step 3 logic: template + note, then formatted with function
        processed_template = background_template + '(Note: Please ignore the reasoning through {gene})'
        context = processed_template.format(gene="[MASKED GENE]", function_name=function_name)
        return context
    
    return None


def _reconstruct_analysis_target(
    gene: str,
    function_name: str,
    analysis_type: str,
    background: Optional[str] = None,
) -> str:
    """Reconstructs the human-readable target used by Step 3."""
    if analysis_type == "gene_vs_function":
        return f"'{gene}' and '{function_name}'"
    if analysis_type == "gene_vs_context":
        if not background:
            raise ValueError("Missing reconstructed background for gene_vs_context")
        return f"the role of '{gene}' in the research context: '{background}'"
    if analysis_type == "function_vs_context":
        if not background:
            raise ValueError("Missing reconstructed background for function_vs_context")
        return f"the role of '{function_name}' in the research context: '{background}'"
    return analysis_type




async def _normalize_and_score_evidence(
    pmid: str,
    gene: str,
    llm_sentences: List[str],
    milvus_manager: MilvusManager,
    embedding_manager: EmbeddingManager,
    score_map: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Maps LLM-generated sentences back to original text in Milvus and attaches scores.
    """
    if not llm_sentences:
        return []
        
    normalized_data = []
    
    try:
        # 1. Embedding
        vectors = await asyncio.to_thread(
            embedding_manager.get_embeddings, llm_sentences, text_type='document'
        )
        if vectors is None or len(vectors) == 0:
            logger.warning("Could not embed evidence candidates for PMID %s / gene %s; omitting unmatched evidence.", pmid, gene)
            return []

        search_vectors = [v.tolist() for v in vectors]
        
        # 2. Reverse Search in Milvus (PMID and gene scoped)
        def quote_filter_value(value: str) -> str:
            return str(value).replace("\\", "\\\\").replace("'", "\\'")

        search_filter = (
            f"pmid == '{quote_filter_value(pmid)}' "
            f"and gene == '{quote_filter_value(gene)}'"
        )
        
        unmatched_count = 0
        for i, vec in enumerate(search_vectors):
            # Search one by one to ensure mapping
            res = milvus_manager.search([vec], search_filter, top_k=1)
            
            if res:
                best_match = res[0]
                match_id = best_match['entity']['sentence_id']
                match_text = best_match['entity']['text']
                # Retrieve Score from Step 2 map
                score = score_map.get(match_id, 0.0) # Default to 0 if missing
                
                normalized_data.append({
                    "text": match_text, # Use standard original text
                    "score": score,
                    "id": match_id,
                    "original_llm": llm_sentences[i] # Keep for debug if needed
                })
            else:
                # Do not present model-generated wording as source evidence.
                unmatched_count += 1

        if unmatched_count:
            logger.warning(
                "No source sentence matched %s of %s evidence candidate(s) for PMID %s / gene %s.",
                unmatched_count,
                len(search_vectors),
                pmid,
                gene,
            )
                
    except Exception as e:
        logger.warning(
            "Evidence normalization failed for PMID %s / gene %s (%s); omitting unmatched evidence.",
            pmid,
            gene,
            type(e).__name__,
        )
        return []

    return normalized_data




def _construct_condensation_messages(
    task_data: Dict[str, Any], 
    function_name: str,
    research_background_template: str,
) -> Optional[List[Dict[str, str]]]:
    """
    [PURE FUNCTION] Constructs the message list for Phase 1 Condensation.
    Used by both Real-time execution and Batch submission.
    """
    pmid = task_data['pmid']
    gene = task_data['gene']
    analysis_type = task_data['analysis_type']
    
    # 1. Background & Context
    messages = [{"role": "system", "content": "You are a rigorous biomedical research assistant who bases reasoning and answers solely on the information provided."}]
    
    reconstructed_background = _reconstruct_background(
        background_template=research_background_template,
        gene=gene,
        function_name=function_name,
        analysis_type=analysis_type
    )
    if reconstructed_background:
        reconstructed_background = reconstructed_background.strip()
    background_text = reconstructed_background
    background_text = f"**The research background described by the user:**\n'''\n{background_text}\n'''\n\n" if background_text else ""
    
    input_text_parts = []
    if task_data.get('title'): input_text_parts.append(f"# Article Title:{task_data['title']}") 
    if task_data.get('abstract_text'): input_text_parts.append(f"# Article Abstract:\n{task_data['abstract_text']}")
    if task_data.get('paragraphs_text'): input_text_parts.append(f"# Highly Relevant Sentences::\n{task_data['paragraphs_text']}")

    input_text_for_llm = "\n\n---\n\n".join(input_text_parts)
    initial_context = f"Article Text:\n'''\n{input_text_for_llm}\n'''\n\n{background_text}Please answer the following questions based solely on the content of the provided text:"

    try:
        # 2. Reconstruct CoT History
        reasoning_trace = json.loads(task_data['reasoning_trace'])
        analysis_target = _reconstruct_analysis_target(
            gene,
            function_name,
            analysis_type,
            reconstructed_background,
        )
        q1_prompt_text = reasoning_trace[0]
        
        full_q1_prompt = f"{initial_context}\n\n**Question 1:**\n{q1_prompt_text}"
        messages.append({"role": "user", "content": full_q1_prompt})
        messages.append({"role": "assistant", "content": reasoning_trace[1]})
        
        for i in range(1, 3):
            if len(reasoning_trace) > i*2+1:
                messages.append({"role": "user", "content": reasoning_trace[i*2]})
                messages.append({"role": "assistant", "content": reasoning_trace[i*2+1]})
        
        # 3. Final Prompt
        task_description = f"Based on the above analysis, extract and summarize the core information related to '{analysis_target}' from the provided scientific text."
        
        summarization_prompt = f"""{task_description}

Your summary must follow a scientific logic structure, which adapts to both experimental studies and review articles:

- If the text reports original experiments: integrate Experimental Methods, Experimental Results, and Experimental Conclusions.
- If the text is a review: integrate the compared experimental approaches across studies, the major reported findings or trends, and any synthesized interpretations, consensus, or debates within the field.

For each part included in the summary, directly quote crucial original sentences from the text as evidence. Use exact original sentences only.

**Output Requirements:**  
Return a *strictly valid* JSON object with the following structure:

{{
  "summary": "A comprehensive scientific summary regarding `{analysis_target}`, covering the methods/approaches used, key findings or observed trends, and the synthesized conclusions, reflecting the scientific reasoning presented in the text.",
  "evidence": [
    "Exact original sentence 1 used as evidence.",
    "Exact original sentence 2 used as evidence.",
    "..."
  ]
}}
        """
        messages.append({"role": "user", "content": summarization_prompt})
        return messages

    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as e:
        logger.warning("Could not prepare Step 4 prompts for PMID %s (%s).", pmid, type(e).__name__)
        return None

# =================================================================
# === 2. Phase 1: Per-Document Information Condensing ===
# =================================================================

async def _run_single_document_condensation_realtime(
    llm_manager,
    semaphore,
    task_data,
    function_name: str,
    research_background_template: str,
    on_started: Optional[Callable[[], None]] = None,
):
    async with semaphore:
        if on_started is not None:
            on_started()
        messages = _construct_condensation_messages(
            task_data,
            function_name,
            research_background_template,
        )
        if not messages: return {"status": "failed", **task_data}
        
        for attempt in range(MAX_RETRIES):
            try:
                resp = await asyncio.to_thread(llm_manager.call, messages_list=messages, temperature=1, response_format={"type": "json_object"})
                content = get_llm_response_content(resp)
                if not content: raise ValueError("Empty response")

                summary, evidence = _parse_condensation_output(content)

                return {"status": "success", "condensed_summary": summary, "evidence_sentences": evidence, **task_data}
            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    logger.warning(
                        "Step 4 document condensation failed for PMID %s / gene %s / analysis %s "
                        "after %s attempt(s) (%s).",
                        task_data.get("pmid", "unknown"),
                        task_data.get("gene", "unknown"),
                        task_data.get("analysis_type", "unknown"),
                        MAX_RETRIES,
                        type(exc).__name__,
                    )
                    return {"status": "failed", "error": type(exc).__name__, **task_data}
                logger.debug(
                    "Retrying Step 4 document condensation for PMID %s / gene %s "
                    "(attempt %s/%s; %s).",
                    task_data.get("pmid", "unknown"),
                    task_data.get("gene", "unknown"),
                    attempt + 1,
                    MAX_RETRIES,
                    type(exc).__name__,
                )
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    return {"status": "failed", "error": "Retry limit reached", **task_data}

# =================================================================
# === 3. Batch 提交逻辑 (Phase 1 Submit) ===
# =================================================================

async def _submit_batch_job(unique_tasks, analysis_data, services, batch_dir, task_fingerprint):
    """生成 JSONL 并提交 Batch 任务。"""
    batch_manager: BatchManager = services["batch_manager"]
    llm_manager: LLMManager = services["llm_manager"]
    
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_path = batch_dir / "step4_batch_input.jsonl"
    
    logger.debug("Preparing a Step 4 Batch request for %s task(s).", len(unique_tasks))
    
    count = 0
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for key, task in unique_tasks.items():
            msgs = _construct_condensation_messages(task, analysis_data.function_name, analysis_data.research_background_template)
            if not msgs: continue
            
            custom_id = f"{task['pmid']}|{task['gene']}|{task['analysis_type']}"
            req = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": llm_manager.model_name,
                    "messages": msgs,
                    "temperature": 1,
                    "response_format": {"type": "json_object"}
                }
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            count += 1
            
    if count == 0: return None

    logger.debug("Uploading the Step 4 Batch input file with %s request(s).", count)
    fid = await asyncio.to_thread(batch_manager.upload_file, jsonl_path)
    job_id = await asyncio.to_thread(batch_manager.create_batch_job, fid, description=f"Step4_{analysis_data.function_name}")
    
    # 保存元数据
    meta_path = batch_dir / "step4_batch_meta.json"
    with open(meta_path, 'w') as f:
        json.dump({
            "batch_id": job_id,
            "file_id": fid,
            "status": "submitted",
            "task_fingerprint": task_fingerprint,
            "last_activity_at": time.time(),
        }, f)
        
    logger.info("Submitted Step 4 Batch job %s.", job_id)
    return job_id

async def _process_batch_results(batch_id, analysis_data, services, batch_dir: Path):
    """下载、本地备份并解析 Batch 结果。"""
    batch_manager = services["batch_manager"]
    logger.info("Downloading results for Step 4 Batch job %s.", batch_id)
    
    details = await asyncio.to_thread(batch_manager.get_batch_details, batch_id)
    if not details.output_file_id:
        logger.error("Step 4 Batch completed without an output file.")
        return []
        
    text = await asyncio.to_thread(batch_manager.download_results, details.output_file_id)
    
    # [新增] 本地备份结果文件 (类似 Step 3 的做法)
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    output_dump_path = batch_dir / "step4_batch_output.jsonl"
    try:
        with open(output_dump_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.debug("Saved the raw Step 4 Batch output to '%s'.", output_dump_path)
    except Exception as e:
        logger.warning(
            "Could not save the Step 4 Batch output backup '%s' (%s).",
            output_dump_path,
            type(e).__name__,
        )

    results = []
    for item in batch_manager.parse_batch_output_to_dict(text):
        try:
            pmid, gene, ana_type = item["custom_id"].split("|", 2)
        except (KeyError, ValueError, AttributeError) as exc:
            logger.warning("Ignoring a malformed Step 4 Batch result identity (%s).", type(exc).__name__)
            continue

        result = {"pmid": pmid, "gene": gene, "analysis_type": ana_type}
        if item.get("response", {}).get("status_code") != 200:
            results.append({"status": "failed", "error": "Batch response status was not 200", **result})
            continue

        try:
            content = item["response"]["body"]["choices"][0]["message"]["content"]
            summary, evidence = _parse_condensation_output(content)
            results.append({
                "status": "success",
                "condensed_summary": summary,
                "evidence_sentences": evidence,
                **result,
            })
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            results.append({"status": "failed", "error": str(exc), **result})
    failed_count = sum(result.get("status") != "success" for result in results)
    logger.debug("Step 4 Batch returned %s result(s) for %s task(s).", len(results), len(unique_tasks))
    if failed_count:
        logger.warning("Step 4 Batch returned %s failed or malformed task result(s).", failed_count)
    return results

async def _wait_for_batch_completion(
    batch_id: str,
    batch_manager: BatchManager,
    check_interval: int = 60,
    idle_timeout_seconds: int = BATCH_INACTIVITY_TIMEOUT_SECONDS,
    activity_state_file: Optional[Path] = None,
) -> str:
    """Wait until completion, terminal failure, or 24 hours without observable progress."""
    logger.info("Waiting for Step 4 Batch job %s.", batch_id)
    activity_state = {}
    if activity_state_file and activity_state_file.exists():
        try:
            activity_state = json.loads(activity_state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            activity_state = {}
    if not isinstance(activity_state, dict):
        activity_state = {}
    try:
        last_activity_at = float(activity_state.get("last_activity_at", time.time()))
    except (TypeError, ValueError):
        activity_state.pop("last_activity_at", None)
        last_activity_at = time.time()
    last_signature = activity_state.get("last_activity_signature")

    def save_activity(signature, activity_at):
        if not activity_state_file:
            return
        activity_state["last_activity_at"] = activity_at
        activity_state["last_activity_signature"] = signature
        try:
            activity_state_file.write_text(
                json.dumps(activity_state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning(
                "Could not persist Step 4 Batch activity state '%s' (%s).",
                activity_state_file,
                type(exc).__name__,
            )

    if "last_activity_at" not in activity_state:
        save_activity(last_signature, last_activity_at)
    
    while True:
        try:
            if hasattr(batch_manager, "check_batch_activity"):
                status, request_counts = await asyncio.to_thread(
                    batch_manager.check_batch_activity, batch_id
                )
            else:
                status = await asyncio.to_thread(batch_manager.check_batch_status, batch_id)
                request_counts = None
            signature = [status, list(request_counts) if request_counts is not None else None]
            now = time.time()
            if signature != last_signature:
                last_signature = signature
                last_activity_at = now
                save_activity(signature, last_activity_at)
            elapsed = max(0, int(now - last_activity_at))
            
            if status == "completed":
                logger.info("Step 4 Batch job %s completed.", batch_id)
                return status
            elif status in ["failed", "expired", "cancelled"]:
                logger.warning("Step 4 Batch job %s ended with status '%s'.", batch_id, status)
                return status
            else:
                if elapsed >= idle_timeout_seconds:
                    logger.warning(
                        "Step 4 Batch job %s made no progress for %.1f hours; switching to realtime.",
                        batch_id,
                        elapsed / 3600,
                    )
                    return "idle_timeout"
                logger.debug(
                    "Step 4 Batch job %s status is '%s'; idle for %.1f hours.",
                    batch_id,
                    status,
                    elapsed / 3600,
                )
                await asyncio.sleep(min(check_interval, max(1, idle_timeout_seconds - elapsed)))
        except Exception as e:
            logger.debug(
                "Could not check Step 4 Batch job %s status (%s); retrying.",
                batch_id,
                type(e).__name__,
            )
            elapsed = max(0, time.time() - last_activity_at)
            if elapsed >= idle_timeout_seconds:
                logger.warning(
                    "Step 4 Batch status checks made no progress for %.1f hours; switching to realtime.",
                    elapsed / 3600,
                )
                return "idle_timeout"
            await asyncio.sleep(min(check_interval, max(1, idle_timeout_seconds - elapsed)))


async def _run_step_4_realtime_tasks(
    unique_tasks,
    llm_manager,
    function_name: str,
    research_background_template: str,
) -> List[Dict[str, Any]]:
    """Condense article evidence in realtime and show active gene/PMID tasks."""
    phase1_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    task_list = list(unique_tasks.values())
    active_tasks = []

    with tqdm(
        total=len(task_list),
        desc="Step 4 article synthesis",
        unit="summary task",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}",
        disable=not progress_enabled(logger),
    ) as progress:
        def refresh_active_tasks() -> None:
            progress.set_postfix_str(active_tasks[-1] if active_tasks else "")

        async def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
            gene = str(task.get("gene", "unknown"))
            pmid = str(task.get("pmid", "unknown"))
            analysis_type = str(task.get("analysis_type", ""))
            relationship = RELATIONSHIP_LABELS.get(analysis_type, analysis_type or "unknown")
            label = f"{gene} | PMID {pmid} | {relationship}"
            started = False

            def mark_started() -> None:
                nonlocal started
                started = True
                active_tasks.append(label)
                refresh_active_tasks()

            try:
                return await _run_single_document_condensation_realtime(
                    llm_manager,
                    phase1_semaphore,
                    task,
                    function_name,
                    research_background_template,
                    on_started=mark_started,
                )
            finally:
                if started:
                    active_tasks.remove(label)
                    progress.update(1)
                    refresh_active_tasks()

        tasks = [run_task(task) for task in task_list]
        return [await future for future in asyncio.as_completed(tasks)]


async def _switch_step_4_to_realtime(
    batch_manager: BatchManager,
    batch_id: str,
    reason: str,
    fallback_marker: Path,
    batch_meta_file: Path,
) -> None:
    if reason != "expired":
        try:
            await asyncio.to_thread(batch_manager.cancel_batch_job, batch_id)
            logger.debug("Requested cancellation of idle Step 4 Batch job %s.", batch_id)
        except Exception as exc:
            logger.warning(
                "Could not cancel Step 4 Batch job %s before realtime fallback (%s).",
                batch_id,
                type(exc).__name__,
            )
    fallback_marker.write_text(
        json.dumps({"batch_id": batch_id, "reason": reason}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if batch_meta_file.exists():
        try:
            metadata = json.loads(batch_meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
        if metadata.get("batch_id") == batch_id:
            batch_meta_file.unlink()
    if reason == "idle_timeout":
        message = "Step 4 Batch had no progress for 24 hours; using realtime calls."
    else:
        message = f"Step 4 Batch ended with {reason}; using realtime calls."
    raise BatchInactivityTimeout(message)


# =================================================================
# === 3. Phase 2: Cross-Document Synthesis ===
# =================================================================

async def _run_cross_document_synthesis(
    llm_manager: LLMManager,
    semaphore: asyncio.Semaphore,
    gene: str,
    analysis_type: str,
    condensed_summaries: List[Dict[str, Any]],
    function_name: str,
    research_background_template: str,
    on_started: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """
    Performs a multi-turn CoT to synthesize information from multiple documents for a specific gene.
    """
    async with semaphore:
        if on_started is not None:
            on_started()
        background_text = _reconstruct_background(
            background_template=research_background_template,
            gene=gene,
            function_name=function_name,
            analysis_type=analysis_type
        )
        if background_text:
            background_text = background_text.strip()
        analysis_target = _reconstruct_analysis_target(
            gene,
            function_name,
            analysis_type,
            background_text,
        )
        input_context_parts = [
            f"Below are summaries extracted from multiple scientific articles regarding the analysis target: Each record includes the source PMID, a core summary, and key evidence sentences."
        ]
        for summary in condensed_summaries:
            ev_list = summary.get('evidence_sentences', [])
            if not isinstance(ev_list, list): 
                ev_list = []
            evidence_str = "\n".join([f"- {s}" for s in ev_list]) if ev_list else "Relevant evidence may be included in the corresponding summary"
            
            input_context_parts.append(
                f"---\nPMID: {summary['pmid']}\nSummary: {summary['condensed_summary']}\nEvidence:\n{evidence_str}"
            )
        
        initial_context = "\n".join(input_context_parts)
        background_text = f"**The research background described by the user:**\n'''\n{background_text}\n'''\n\n" if background_text else ""

        synthesis_questions = [
            f"""
From all provided abstracts, identify the key scientific information directly related to {analysis_target}. For each study, extract:
1) Experimental Methods (what was done),
2) Experimental Results (what was observed), and
3) Experimental Conclusions (the interpretation supported by the data).
When extracting evidence, classify each study's validation strength based on its experimental system (e.g., Human > Primate > Large animal > Mouse > Zebrafish > Drosophila > C. elegans > Cell line). Clearly label each extracted piece with its evidence tier and the PMID.
""",
            f"Synthesize the cross-study common findings and core mechanisms directly related to {analysis_target}. In this synthesis, weigh evidence based on validation tier: higher-tier evidence (e.g., patient or primate data) should have greater influence on the summary's conclusions than lower-tier models. Also identify any conflicts, disagreements, or divergent interpretations among the studies.",
            f"Integrate the above into a coherent scientific summary. The summary must explicitly reflect scientific reasoning: how the methods lead to results, and how the results justify the conclusions.\nSupport each statement with one or more PMIDs in the format [PMID: XXXXX], listed separately when multiple sources apply.(for example, [PMID: 12345], [PMID: 98765])",
            "Wait, please review this summary. Have you over summarized some of the information in it?",
            f"Provide a revised final summary of {analysis_target}. Use Markdown format and include only the scientific content. The summary should focus on articulating the scientific understanding of {analysis_target}, integrating evidence from different studies in a natural way. Where studies support or differ from one another, reflect this within the narrative without explicitly emphasizing the comparison itself. Ensure all PMIDs are correctly formatted and support the statements."
        ]
        
        messages = [{"role": "system", "content": "You are a professional science writer. Please synthesize information from multiple provided sources into a coherent narrative on a specific topic, using only the given materials—without adding external knowledge—and properly cite each source."}]
        
        current_context = f"**Articles:**\n'''\n{initial_context}\n'''\n\n{background_text}Based on the information provided, please answer the following questions:"

        for i, question in enumerate(synthesis_questions):
            if i == 0:
                # 第一轮：Context + Q1 -> 开启缓存
                full_prompt = f"{current_context}\n\n**Question {i+1}:**\n{question}"
                message_content = llm_manager.create_cached_content(full_prompt)
            else:
                # 后续轮次：仅发送问题
                message_content = f"**Question {i+1}:**\n{question}"
            messages.append({"role": "user", "content": message_content})

            response_data = await asyncio.to_thread(llm_manager.call, messages_list=messages, temperature=1)
            if response_data is None:
                logger.warning(
                    "Step 4 synthesis failed for gene '%s', analysis '%s', substep %s.",
                    gene,
                    analysis_type,
                    i + 1,
                )
                return {"status": "failed", "gene": gene, "analysis_type": analysis_type, "final_summary": ""}
            step_response_content = get_llm_response_content(response_data)

            messages.append({"role": "assistant", "content": step_response_content})

        final_summary = messages[-1].get("content", "Failed to extract final summary.")
        source_pmids = [s['pmid'] for s in condensed_summaries]

        return {
            "status": "success",
            "gene": gene,
            "analysis_type": analysis_type,
            "final_summary": final_summary.strip(),
            "source_pmids": source_pmids,
            "cot_trace": messages,
        }





async def _run_global_context_synthesis(
    llm_manager: LLMManager,
    summaries: List[str],
    function_name: str,
    background_template: str,
) -> str:
    if not summaries: return ""
    if len(summaries) == 1: return summaries[0]
    
    combined_text = "".join([f"\n[Source Summary {i+1}]:\n{s}\n" for i, s in enumerate(summaries)])
    clean_context = background_template.replace("{gene}", "Target Gene").replace("{function_name}", function_name)
    
    prompt = f"""
We have conducted a literature analysis on the relationship between: **"{function_name}"** and **"{clean_context}"**.
Based on the the partial summaries below (from different gene searches) below, synthesize a concise research background describing the relationship between **"{function_name}"** and **"{clean_context}"**.

**Requirements:**
- Focus on how this relationship has been revealed through scientific investigation, allowing observations, experimental findings, and inferred mechanisms to appear naturally in the narrative.
- Integrate overlapping content into a coherent explanation centered on the function–context relationship, not on individual genes.
- Let differences in experimental systems implicitly inform the narrative, without explicitly ranking or labeling evidence.
- Preserve all inline citations exactly as provided (e.g., [PMID:12345678]).

**Formatting:**
- Provide a suitable and relatively short title based on the content.
- Write in Markdown with natural paragraph breaks.

**Source Summaries:**
{combined_text}
"""
    messages = [{"role": "system", "content": "You are a professional science writer. Please synthesize information from multiple provided sources into a coherent narrative on a specific topic, using only the given materials—without adding external knowledge—and properly cite each source."}, {"role": "user", "content": prompt}]
    try:
        resp = await asyncio.to_thread(llm_manager.call, messages_list=messages, temperature=1)
        return get_llm_response_content(resp).strip()
    except Exception:
        return "\n\n".join(summaries)


        

async def run_synthesis_step(
    analysis_data: AnalysisData,
    services: Dict[str, Any],
    batch_run_id: Optional[str] = None,
) -> AnalysisData:

    if analysis_data.relevance_filter_status in {"failed", "partial", "no_candidates", "stale"}:
        raise RuntimeError(
            "Step 4 requires complete Step 3 results; "
            f"received status={analysis_data.relevance_filter_status!r}."
        )
    
    llm_manager = services["llm_manager"]
    embedding_manager = services["embedding_manager"]
    batch_manager = services["batch_manager"]
    llm_model = llm_manager.model_name
    target_gene_keys = {
        str(gene).strip().casefold()
        for gene in analysis_data.genes
        if str(gene).strip() and str(gene).strip().casefold() != "global"
    }
    is_multi_gene_analysis = len(target_gene_keys) > 1
    
    analysis_dir = Path(analysis_data.project_folder) / f"analysis_{analysis_directory_name(analysis_data.function_name)}"
    db_path = analysis_dir / "step2_embeddings.db"
    
    # 统一路径管理：所有 Batch 相关文件放入 batch_files
    batch_files_dir = analysis_dir / 'batch_files'
    if batch_run_id:
        batch_files_dir = batch_files_dir / batch_run_id
    batch_files_dir.mkdir(parents=True, exist_ok=True)
    batch_meta_file = batch_files_dir / "step4_batch_meta.json"
    
    use_batch = BATCH_API_ENABLED and llm_model in BATCH_SUPPORTED_MODELS
    # --- Load Data ---
    milvus_reader = MilvusManager(
        db_path=str(db_path),
        collection_name="biomed_sentences",
        dimension=embedding_manager.dimensions,
    )
    try:
        if not milvus_reader.client.has_collection(milvus_reader.llm_analysis_collection_name):
             raise RuntimeError("Step 3 result collection is missing from the analysis database.")
        active_genes = {str(gene).casefold() for gene in analysis_data.genes}
        active_pairs = {
            (str(row.get("PMID", "")), str(row.get("Gene", "")))
            for _, row in analysis_data.recalled_sentences_df.iterrows()
        }
        step3_recs = [
            row for row in milvus_reader.load_llm_analysis_results()
            if str(row.get("result", "")).strip().casefold() == "yes"
            and str(row.get("gene", "")).casefold() in active_genes
            and (str(row.get("pmid", "")), str(row.get("gene", ""))) in active_pairs
        ]
    finally:
        milvus_reader.close()

    # --- Prepare Tasks ---
    titles_map = analysis_data.raw_papers_df.set_index('PMID')['title'].to_dict() if not analysis_data.raw_papers_df.empty else {}
    unique_tasks = {}
    task_mapping = defaultdict(list)
    for r in step3_recs:
        r['title'] = titles_map.get(r['pmid'])
        key = (str(r['pmid']), str(r['gene']), str(r['analysis_type']))
        task_mapping[key].append(r)
        if key not in unique_tasks: unique_tasks[key] = r

    if not unique_tasks:
        logger.warning(
            "Step 4 LLM-Guided evidence synthesis has no article-level evidence summaries to prepare for target gene(s): %s.",
            ", ".join(analysis_data.genes),
        )
        raise RuntimeError(
            "Step 4 produced no article-level literature summaries because no literature passed "
            "Step 3 relevance filtering. Stopping before Steps 5 and 6."
        )

    article_summary_counts = defaultdict(int)
    for task in unique_tasks.values():
        article_summary_counts[str(task.get("gene", "unknown"))] += 1
    article_summary_breakdown = "; ".join(
        f"{gene}: {article_summary_counts.get(gene, 0)}"
        for gene in analysis_data.genes
    )
    task_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model_name": llm_model,
                "function_name": analysis_data.function_name,
                "research_background_template": analysis_data.research_background_template,
                "tasks": [
                    {
                        "pmid": pmid,
                        "gene": gene,
                        "analysis_type": analysis_type,
                        "task": task,
                    }
                    for (pmid, gene, analysis_type), task in sorted(unique_tasks.items())
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    fallback_marker = batch_files_dir / f"step4_realtime_fallback_{task_fingerprint[:12]}.json"

    if not use_batch:
        execution_mode = "real-time model calls"
    elif fallback_marker.exists():
        execution_mode = "real-time calls after the previous Batch idle timeout"
    else:
        execution_mode = "Batch API"
    logger.info(
        "Step 4 prepared %s article-level evidence-summary request(s) across %s target gene(s) (%s); using %s.",
        len(unique_tasks),
        len(analysis_data.genes),
        article_summary_breakdown,
        execution_mode,
    )

    # =========================================================
    # === Phase 1: Execution (Auto-Retry Batch Logic) ===
    # =========================================================
    
    raw_phase1_results = []
    
    if use_batch and not fallback_marker.exists():
        try:
            batch_id = None

            # Resume only a job submitted for the same exact task set.
            if batch_meta_file.exists():
                try:
                    meta = json.loads(batch_meta_file.read_text(encoding="utf-8"))
                    batch_id = meta.get("batch_id")
                except (OSError, json.JSONDecodeError, AttributeError):
                    meta = {}
                    batch_id = None

                if batch_id and meta.get("task_fingerprint") != task_fingerprint:
                    logger.warning("Existing Step 4 Batch job belongs to a different task set; replacing it.")
                    try:
                        await asyncio.to_thread(batch_manager.cancel_batch_job, batch_id)
                    except Exception as exc:
                        logger.warning(
                            "Could not cancel stale Step 4 Batch job %s (%s).",
                            batch_id,
                            type(exc).__name__,
                        )
                    batch_meta_file.unlink(missing_ok=True)
                    batch_id = None

                if batch_id:
                    try:
                        status = await asyncio.to_thread(batch_manager.check_batch_status, batch_id)
                    except Exception as exc:
                        # Keep the known ID and let the bounded polling loop retry status checks.
                        logger.warning(
                            "Could not inspect existing Step 4 Batch job %s (%s).",
                            batch_id,
                            type(exc).__name__,
                        )
                    else:
                        if status == "expired":
                            await _switch_step_4_to_realtime(
                                batch_manager, batch_id, status, fallback_marker, batch_meta_file
                            )
                        if status in ["failed", "cancelled"]:
                            logger.warning("Previous Step 4 Batch job %s was %s; submitting a replacement.", batch_id, status)
                            batch_meta_file.unlink(missing_ok=True)
                            batch_id = None

            if not batch_id:
                batch_id = await _submit_batch_job(
                    unique_tasks, analysis_data, services, batch_files_dir, task_fingerprint
                )

            if batch_id:
                final_status = await _wait_for_batch_completion(
                    batch_id,
                    batch_manager,
                    check_interval=30,
                    activity_state_file=batch_meta_file,
                )
                if final_status == "completed":
                    raw_phase1_results = await _process_batch_results(
                        batch_id, analysis_data, services, batch_files_dir
                    )
                elif final_status in {"idle_timeout", "expired"}:
                    await _switch_step_4_to_realtime(
                        batch_manager, batch_id, final_status, fallback_marker, batch_meta_file
                    )
                else:
                    raise RuntimeError(f"Step 4 Batch job failed with status {final_status!r}.")
        except BatchInactivityTimeout as exc:
            logger.warning("%s Running the Step 4 article-level summaries through realtime model calls.", exc)
            raw_phase1_results = await _run_step_4_realtime_tasks(
                unique_tasks,
                llm_manager,
                analysis_data.function_name,
                analysis_data.research_background_template,
            )
    elif use_batch:
        raw_phase1_results = await _run_step_4_realtime_tasks(
            unique_tasks,
            llm_manager,
            analysis_data.function_name,
            analysis_data.research_background_template,
        )
    else:
        # Real-time Fallback
        raw_phase1_results = await _run_step_4_realtime_tasks(
            unique_tasks,
            llm_manager,
            analysis_data.function_name,
            analysis_data.research_background_template,
        )

    # =========================================================
    # === Phase 2: Normalization & Synthesis ===
    # =========================================================
    
    expected_phase1_keys = set(unique_tasks)
    phase1_outcomes = defaultdict(list)
    unexpected_phase1_keys = set()
    for result in raw_phase1_results:
        key = (
            str(result.get("pmid", "")),
            str(result.get("gene", "")),
            str(result.get("analysis_type", "")),
        )
        if key in expected_phase1_keys:
            phase1_outcomes[key].append(result.get("status"))
        else:
            unexpected_phase1_keys.add(key)

    failed_phase1_keys = {
        key for key in expected_phase1_keys
        if phase1_outcomes.get(key) != ["success"]
    }
    phase1_successes_by_gene = defaultdict(int)
    for result in raw_phase1_results:
        if result.get("status") == "success":
            phase1_successes_by_gene[str(result.get("gene", "unknown"))] += 1
    phase1_breakdown = "; ".join(
        f"{gene}: {phase1_successes_by_gene.get(gene, 0)}/{article_summary_counts.get(gene, 0)}"
        for gene in analysis_data.genes
    )
    logger.info(
        "Step 4 article-level evidence summaries succeeded for %s of %s gene-PMID-relationship request(s): %s.",
        sum(phase1_successes_by_gene.values()),
        len(expected_phase1_keys),
        phase1_breakdown,
    )
    if failed_phase1_keys or unexpected_phase1_keys:
        raise RuntimeError(
            "Step 4 document condensation did not complete cleanly: "
            f"{len(failed_phase1_keys)} of {len(expected_phase1_keys)} expected tasks failed or were missing; "
            f"{len(unexpected_phase1_keys)} unexpected result keys were returned."
        )
    # 1. Normalization
    score_map = {}
    if not analysis_data.recalled_sentences_df.empty:
        score_map = dict(zip(analysis_data.recalled_sentences_df['sentence_id'], analysis_data.recalled_sentences_df['cosine_similarity']))

    milvus_searcher = MilvusManager(db_path=str(db_path), collection_name="biomed_sentences", dimension=embedding_manager.dimensions)
    successful_condensations = []
    
    results_map = {}
    for res in raw_phase1_results:
        if res.get('status') == 'success':
            key = (str(res['pmid']), str(res['gene']), str(res['analysis_type']))
            results_map[key] = res
            
    for unique_key, original_rows in task_mapping.items():
        if unique_key in results_map:
            base = results_map[unique_key]
            for row in original_rows:
                rep = base.copy()
                rep['gene'] = str(row['gene'])
                objs = await _normalize_and_score_evidence(
                    str(base['pmid']),
                    rep['gene'],
                    base.get('evidence_sentences', []),
                    milvus_searcher,
                    embedding_manager,
                    score_map,
                )
                rep['evidence_sentences'] = [obj['text'] for obj in objs]
                rep['evidence_objects'] = objs
                successful_condensations.append(rep)
    milvus_searcher.close()

    # 2. Synthesis (Real-time)
    successful_synthesis = []
    if successful_condensations:
        df = pd.DataFrame(successful_condensations)
        phase2_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        grouped_inputs = list(df.groupby(['gene', 'analysis_type']))
        expected_synthesis_groups = {
            (str(name[0]), str(name[1])) for name, _ in grouped_inputs
        }
        active_synthesis_groups = []

        async def run_group_synthesis(name, group):
            gene, analysis_type = str(name[0]), str(name[1])
            relationship = RELATIONSHIP_LABELS.get(analysis_type, analysis_type)
            label = f"{gene} | {relationship}"
            started = False

            def mark_started() -> None:
                nonlocal started
                started = True
                active_synthesis_groups.append(label)
                synthesis_progress.set_postfix_str(active_synthesis_groups[-1])

            try:
                return await _run_cross_document_synthesis(
                    llm_manager,
                    phase2_semaphore,
                    gene,
                    analysis_type,
                    group.to_dict('records'),
                    analysis_data.function_name,
                    analysis_data.research_background_template,
                    on_started=mark_started,
                )
            except Exception as exc:
                logger.warning(
                    "Step 4 cross-document synthesis failed for gene %s / analysis %s (%s).",
                    gene,
                    analysis_type,
                    type(exc).__name__,
                )
                return {
                    "status": "failed",
                    "gene": gene,
                    "analysis_type": analysis_type,
                    "error": str(exc),
                }
            finally:
                if started:
                    active_synthesis_groups.remove(label)
                    synthesis_progress.update(1)
                    synthesis_progress.set_postfix_str(
                        active_synthesis_groups[-1] if active_synthesis_groups else "",
                    )

        tasks_p2 = [run_group_synthesis(name, group) for name, group in grouped_inputs]
        with tqdm(
            total=len(tasks_p2),
            desc="Step 4 gene synthesis",
            unit="summary",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}",
            disable=not progress_enabled(logger),
        ) as synthesis_progress:
            phase2_results = [
                await future
                for future in asyncio.as_completed(tasks_p2)
            ]
        successful_synthesis = [
            result for result in phase2_results
            if result.get('status') == 'success'
            and isinstance(result.get('final_summary'), str)
            and result['final_summary'].strip()
        ]
        successful_synthesis_groups = {
            (str(result.get('gene', '')), str(result.get('analysis_type', '')))
            for result in successful_synthesis
        }
        failed_synthesis_groups = expected_synthesis_groups - successful_synthesis_groups
        successful_groups_by_gene = defaultdict(list)
        for gene, analysis_type in successful_synthesis_groups:
            successful_groups_by_gene[gene].append(
                RELATIONSHIP_LABELS.get(analysis_type, analysis_type)
            )
        group_breakdown = "; ".join(
            f"{gene}: {', '.join(sorted(successful_groups_by_gene.get(gene, []))) or 'no relationship summaries'}"
            for gene in analysis_data.genes
        )
        logger.info(
            "Step 4 cross-article evidence synthesis produced final summaries for %s of %s gene-relationship group(s): %s.",
            len(successful_synthesis_groups),
            len(expected_synthesis_groups),
            group_breakdown,
        )
        if failed_synthesis_groups:
            for gene, analysis_type in sorted(failed_synthesis_groups):
                logger.warning(
                    "Step 4 cross-document synthesis has no final summary for gene %s / analysis %s.",
                    gene,
                    analysis_type,
                )
            raise RuntimeError(
                "Step 4 cross-document synthesis was incomplete: "
                f"{len(failed_synthesis_groups)} of {len(expected_synthesis_groups)} groups failed or returned no summary."
            )

    # =========================================================
    # === Saving Results ===
    # =========================================================
    # 1. Add Per-Document Records
    all_final_records = []
    evidence_agg = defaultdict(lambda: defaultdict(list))

    # 1. Add Per-Document Records (保持不变)
    for r in successful_condensations:
        k = (r['gene'], r['analysis_type'])
        evidence_agg[k][r['pmid']].extend(r.get('evidence_objects', []))
        ev_str = "; ".join(f'"{_clean_csv_text(o["text"])}"' for o in r.get('evidence_objects', []))
        all_final_records.append({"Gene": r['gene'], "Analysis_Type": r['analysis_type'], "Summary_Level": "per_document", "PMID": str(r['pmid']), "Summary_Text": r.get('condensed_summary', ''), "Evidence": ev_str, "_evidence_objects": r.get('evidence_objects', [])})
    
    # 2. Add Individual Synthesis & Collect Data for Global Synthesis
    fc_summaries_for_global = [] 
    # [逻辑关键点]: 结构为 { PMID_str: { 句子文本: max_score_float } }
    fc_global_evidence_pool = defaultdict(dict) 
    
    seen_groups = set()
    for r in successful_synthesis:
        k = (r['gene'], r['analysis_type'])
        seen_groups.add(k)
        
        # --- 原有的单个基因 Evidence JSON 生成 (保持不变) ---
        ev_json = {}
        pmids = []
        for pmid, objs in evidence_agg.get(k, {}).items():
            pmids.append(str(pmid))
            seen_ids = set()
            formatted = []
            for o in objs:
                if o['id'] and o['id'] not in seen_ids:
                    seen_ids.add(o['id'])
                    s = f"{o['score']:.2f}" if o.get('score') is not None else "N/A"
                    formatted.append(f"{_clean_csv_text(o['text'])} (score:{s})")
                elif not o['id']: formatted.append(_clean_csv_text(o['text']))
            if formatted: ev_json[str(pmid)] = formatted
        
        final_text = r.get('final_summary', '')
        
        # --- [NEW] 收集 Global 所需数据 ---
        if is_multi_gene_analysis and r['analysis_type'] == 'function_vs_context' and final_text:
            # A. 收集文本
            fc_summaries_for_global.append(final_text)
            
            # B. 收集证据 (去重并保留最高分)
            current_gene_evidence_map = evidence_agg.get(k, {})
            for pmid, objs in current_gene_evidence_map.items():
                pmid_str = str(pmid)
                for o in objs:
                    raw_text = _clean_csv_text(o['text'])
                    if not raw_text: continue
                    
                    # 获取当前分数 (如果None默认为0)
                    curr_score = o.get('score', 0.0) 
                    if curr_score is None: curr_score = 0.0
                    
                    # [核心逻辑]: 如果句子已存在，保留分数较高者；不存在则直接存入
                    if raw_text not in fc_global_evidence_pool[pmid_str]:
                        fc_global_evidence_pool[pmid_str][raw_text] = curr_score
                    else:
                        if curr_score > fc_global_evidence_pool[pmid_str][raw_text]:
                            fc_global_evidence_pool[pmid_str][raw_text] = curr_score

        # 保存当前基因记录
        all_final_records.append({"Gene": r['gene'], "Analysis_Type": r['analysis_type'], "Summary_Level": "final_synthesis", "PMID": "", "Summary_Text": final_text, "Evidence": "; ".join(sorted(list(set(pmids)))), "Evidence_JSON": json.dumps(ev_json, ensure_ascii=False)})

    # 3. [NEW] Run Global Context Synthesis
    if is_multi_gene_analysis and fc_summaries_for_global:
        logger.info(
            "Step 4 is generating the global function-context synthesis from %s source summary(ies).",
            len(fc_summaries_for_global),
        )
        
        # A. 生成 Global 文本
        global_fc_text = await _run_global_context_synthesis(
            llm_manager, fc_summaries_for_global, analysis_data.function_name, analysis_data.research_background_template
        )
        
        # B. 反查引用并在证据池中提取最高分句子
        referenced_pmids = _extract_pmids_from_text(global_fc_text)
        global_ev_json_dict = {}
        valid_pmid_list = []
        
        for pmid in referenced_pmids:
            if pmid in fc_global_evidence_pool:
                valid_pmid_list.append(pmid)
                formatted_sentences = []
                # 遍历去重后的 {文本:最高分} 字典
                for txt, max_score in fc_global_evidence_pool[pmid].items():
                    s_str = f"{max_score:.2f}"
                    formatted_sentences.append(f"{txt} (score:{s_str})")
                
                # 排序保证确定性
                global_ev_json_dict[pmid] = sorted(formatted_sentences)
        
        # C. 添加 GLOBAL 记录
        all_final_records.append({
            "Gene": "GLOBAL",
            "Analysis_Type": "function_vs_context",
            "Summary_Level": "final_synthesis",
            "PMID": "",
            "Summary_Text": global_fc_text,
            "Evidence": "; ".join(valid_pmid_list),
            "Evidence_JSON": json.dumps(global_ev_json_dict, ensure_ascii=False)
        })
    elif not is_multi_gene_analysis:
        logger.info(
            "Step 4 skipped cross-gene function–research-background synthesis for a single target gene."
        )

    
    # 4. Fill Missing
    for gene in analysis_data.genes:
        for at in ["gene_vs_function", "gene_vs_context", "function_vs_context"]:
            if (gene, at) not in seen_groups:
                all_final_records.append({
                    "Gene": gene,
                    "Analysis_Type": at,
                    "Summary_Level": "final_synthesis",
                    "PMID": "",
                    "Summary_Text": "No qualifying literature passed Step 3 relevance filtering.",
                    "Evidence": "",
                    "Evidence_JSON": "{}",
                })

    if not all_final_records: return analysis_data
    
    df_mem = pd.DataFrame(all_final_records)
    df_mem['rank'] = df_mem['Summary_Level'].apply(
        lambda summary_level: 0 if summary_level == 'final_synthesis' else 1
    )
    df_mem = df_mem.sort_values(by=['Gene', 'Analysis_Type', 'rank', 'PMID']).reset_index(drop=True).drop(columns=['rank'])
    df_mem['Evidence'] = df_mem.apply(
        lambda synthesis_row: synthesis_row.get('Evidence_JSON')
        if synthesis_row['Summary_Level'] == 'final_synthesis' and synthesis_row.get('Evidence_JSON')
        else synthesis_row['Evidence'],
        axis=1,
    )
    generated_synthesis_df = df_mem.drop(columns=['Evidence_JSON', '_evidence_objects'], errors='ignore')
    
    milvus_ents = []
    for _, r in generated_synthesis_df.iterrows():
        rid = f"{r['Gene']}_{r['Analysis_Type']}_final" if r['Summary_Level'] == 'final_synthesis' else f"{r.get('PMID')}_{r['Gene']}_{r['Analysis_Type']}"
        milvus_ents.append({"record_id": rid, "pmid": str(r.get('PMID', '')), "gene": r['Gene'], "analysis_type": r['Analysis_Type'], "model_name": llm_model, "summary_level": r['Summary_Level'], "summary_text": r['Summary_Text'], "evidence_text": r['Evidence']})
    
    mw = MilvusManager(db_path=str(db_path), collection_name="biomed_sentences", dimension=embedding_manager.dimensions)
    try:
        mw.upsert_synthesis_batch(milvus_ents)
        persisted_ids = {entity["record_id"] for entity in milvus_ents}
        persisted_rows = [
            row for row in mw.load_synthesis_results()
            if row.get("record_id") in persisted_ids
        ]
    finally:
        mw.close()

    analysis_data.synthesis_df = pd.DataFrame(persisted_rows)
    df_csv = analysis_data.synthesis_df.copy()
    for column in ['Summary_Text', 'Evidence']:
        if column in df_csv.columns:
            df_csv[column] = df_csv[column].fillna('').astype(str).apply(_clean_csv_text)
    AnalysisSaver(analysis_dir).save_synthesis_results(df_csv)

    logger.info(
        "Step 4 saved %s synthesis record(s) to the analysis database.",
        len(analysis_data.synthesis_df),
    )
    return analysis_data
