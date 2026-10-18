# src/pipeline/step_03_relevance_filter.py

import asyncio
import hashlib
import json
import logging
import re
import os
import time
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from collections import defaultdict

# --- Relative imports ---
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
from ..connectors.batch_manager import BatchManager, BATCH_INACTIVITY_TIMEOUT_SECONDS
from ..logging_utils import progress_enabled

logger = logging.getLogger(__name__)

MAX_CONCURRENT_CHECKS = 3
MAX_RETRIES = 3
RETRY_DELAY = 5


class BatchInactivityTimeout(TimeoutError):
    """Signals that Step 3 should stop using Batch and finish through realtime calls."""



REASONING_QUESTIONS = [
        "In the provided text,Is it directly related to '{analysis_target}'? Please indicate all relevant sentences, and if not, honestly answer 'not mentioned'.",
        "How does the text explain the relationship '{analysis_target}'? Please detail your reasoning. If there is no explanation, answer 'No explanation'.",
        "Wait. Have you overinterpreted or inferred the relationship between '{analysis_target}'?",
        "Based on the above analysis,please confirm whether the text is directly related to the association between'{analysis_target}' and provide your explanation for the judgment result."
    ]

# =================================================================
# === 1. Helper Functions (Unchanged) ===
# =================================================================

def _extract_and_parse_json(text: str) -> Optional[Dict[str, Any]]:
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
    return None

def _prepare_analysis_target(
    background_template: str,
    gene: str,
    function_name: str,
    analysis_type: str,
) -> Dict[str, Optional[str]]:
    if analysis_type == "gene_vs_function":
        return {"context": f"'{gene}' and '{function_name}'", "background": None}
    if analysis_type == "gene_vs_context":
        formatted_template = background_template + "(Note: Please ignore the reasoning through {function_name})"
        context = formatted_template.format(gene=gene, function_name="[MASKED FUNCTION]")
        return {
            "context": f"the role of '{gene}' in the research context: '{context}'",
            "background": context,
        }
    if analysis_type == "function_vs_context":
        formatted_template = background_template + "(Note: Please ignore the reasoning through {gene})"
        context = formatted_template.format(gene="[MASKED GENE]", function_name=function_name)
        return {
            "context": f"the role of '{function_name}' in the research context: '{context}'",
            "background": context,
        }
    raise ValueError(f"Unknown analysis type: {analysis_type}")

def _clean_csv_text(text): return str(text).replace('\n', ' ').strip()


def _normalize_relevance_result(value: Any) -> str:
    """Normalize harmless casing/whitespace variation while rejecting ambiguous answers."""
    if value is None:
        return "parse_failed"
    normalized = str(value).strip().casefold().strip(" .!\"'`")
    return normalized if normalized in {"yes", "no"} else "parse_failed"

def _get_input_text_for_pmid(pmid: str, gene: str, analysis_data: AnalysisData, analysis_dir: Path, filtered_sentences_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    meta_df = analysis_data.raw_papers_df
    paper_meta = meta_df[(meta_df['PMID'] == pmid) & (meta_df['Gene'] == gene)]
    if paper_meta.empty: paper_meta = meta_df[meta_df['PMID'] == pmid]
    if paper_meta.empty: return None
    
    paper_info = paper_meta.iloc[0]
    title_value = paper_info.get('title', '')
    title = "" if pd.isna(title_value) else str(title_value)
    abstract_value = paper_info.get('abstract', '')
    abstract = "" if pd.isna(abstract_value) else str(abstract_value)
            
    paper_sentences_df = filtered_sentences_df[(filtered_sentences_df['PMID'] == pmid) & (filtered_sentences_df['Gene'] == gene)].copy()
    paragraphs_text = ""
    if not paper_sentences_df.empty:
        try:
            paper_sentences_df['para_num'] = paper_sentences_df['paragraph_id'].str.extract(r'_par_(\d+)$').astype(int)
            paper_sentences_df['sent_num'] = paper_sentences_df['sentence_id'].str.extract(r'_sen_(\d+)$').astype(int)
            grouped = paper_sentences_df.sort_values(['para_num', 'sent_num']).groupby('paragraph_id')['text'].apply(
                lambda paragraph_sentences: " ".join(paragraph_sentences)
            )
            paragraphs_text = "\n\n".join(grouped)
        except Exception as exc:
            logger.debug(
                "Could not restore sentence order for PMID %s / gene %s (%s).",
                pmid,
                gene,
                type(exc).__name__,
            )

    if not abstract and not paragraphs_text: return None
    return {"abstract_text": abstract, "paragraphs_text": paragraphs_text, "title": title}

# =================================================================
# === 2. Real-time CoT Logic (Original) ===
# =================================================================

async def _run_single_agent_cot(llm_manager, input_text, background, analysis_target, pmid):
    if background==None:
        initial_context = f"Article Text:\n'''\n{input_text}\n'''\n\nPlease answer the following questions based solely on the content of the provided text:"
    else:
        initial_context = f"**Article Text:**\n'''\n{input_text}\n'''\n\n**The research background described by the user:**\n'''\n{background}\n'''\n\nPlease answer the following questions based solely on the content of the provided text:"
    messages = [{"role": "system", "content": "You are a rigorous biomedical research assistant who bases reasoning and answers solely on the information provided by the user, without adding external knowledge or overinterpretation."}]
    
    trace_list = []
    try:
        for i, question_template in enumerate(REASONING_QUESTIONS):
            question = question_template.format(analysis_target=analysis_target)
            if i == 0:
                full_prompt_text = f"{initial_context}\n\n**Question {i+1}:**\n{question}"
                # --- 调用 create_cached_content ---
                # 如果模型支持且文本够长，这里会返回 [{"type": "text", "text": ..., "cache_control": ...}]
                # 否则返回原字符串
                message_content = llm_manager.create_cached_content(full_prompt_text)
            else:
                # 后续轮次无需缓存
                message_content = f"**Question {i+1}:**\n{question}"
            messages.append({"role": "user", "content": message_content})
            
            trace_list.append(question)
            
            response_data = await asyncio.to_thread(llm_manager.call, messages_list=messages, temperature=1)
            if not response_data: raise ConnectionError(f"LLM call failed at step {i+1}")
            
            content = get_llm_response_content(response_data)
            messages.append({"role": "assistant", "content": content})
            trace_list.append(content)
        return {"final_response": messages[-1]["content"], "reasoning_trace_list": trace_list}
    except Exception as e:
        logger.debug("Step 3 reasoning request failed for PMID %s (%s).", pmid, type(e).__name__)
        return None

async def _format_and_validate_relevance(llm_manager, final_cot, analysis_target, background):
    if background==None:
        format_description = f"""The output in JSON format should contain two fields: "result" and "reason". If the text includes statements directly related to "{analysis_target}", set "result" to "yes"; otherwise, set it to "no". Provide a detailed explanation in the "reason" field."""
    else:
        format_description = f"""**Research Background:**{background}\n\nThe output in JSON format should contain two fields: "result" and "reason". If the text includes statements directly related to "{analysis_target}", set "result" to "yes"; otherwise, set it to "no". Provide a detailed explanation in the "reason" field."""
    messages = [
        {"role": "system", "content": "You are a formatting function. You will only output the content in the transformed format and nothing else."},
        {"role": "user", "content": f"**Content**:\n{final_cot}\n**Formatting Requirements**:{format_description}"}
    ]
    resp = await asyncio.to_thread(llm_manager.call, messages_list=messages, temperature=1, response_format={"type": "json_object"})
    return get_llm_response_content(resp)

async def _process_single_agent_task(
    llm_manager: LLMManager,
    semaphore: asyncio.Semaphore,
    abstract: str,
    paragraphs: str,
    background: Optional[str],
    analysis_target: str,
    pmid: str,
    gene: str,
    title: str,
    analysis_type: str,
    on_started: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    async with semaphore:
        if on_started is not None:
            on_started()
        input_text_parts = []
        if title:
           input_text_parts.append(f"# Article Title:{title}") 
        if abstract:
            input_text_parts.append(f"# Article Abstract:\n{abstract}")
        if paragraphs:
            input_text_parts.append(f"# Highly Relevant Sentences:\n{paragraphs}")
        input_text_for_llm = "\n\n---\n\n".join(input_text_parts)
        
        for attempt in range(MAX_RETRIES):
            failure_status = "failed"
            try:
                cot = await _run_single_agent_cot(
                    llm_manager,
                    input_text_for_llm,
                    background,
                    analysis_target,
                    pmid,
                )
                if not cot: raise ValueError("CoT failed")
                
                json_str = await _format_and_validate_relevance(
                    llm_manager,
                    cot["final_response"],
                    analysis_target,
                    background,
                )
                parsed = _extract_and_parse_json(json_str)
                if not parsed:
                    failure_status = "parse_failed"
                    raise ValueError("Could not parse the relevance response as JSON")
                normalized_result = _normalize_relevance_result(parsed.get('result'))
                if normalized_result == "parse_failed":
                    failure_status = "parse_failed"
                    raise ValueError("Relevance response result must be yes or no")
                
                return {
                    "gene": gene, "pmid": pmid, "title": title,
                    "analysis_type": analysis_type, "status": "success",
                    "abstract_text": abstract, "paragraphs_text": paragraphs,
                    "reasoning_trace_list": cot["reasoning_trace_list"],
                    "result": normalized_result, "json_reason": str(parsed.get('reason', ''))
                }
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.warning(
                        "Step 3 relevance task failed for PMID %s / gene %s / analysis %s "
                        "after %s attempt(s) (%s; status=%s).",
                        pmid,
                        gene,
                        analysis_type,
                        MAX_RETRIES,
                        type(e).__name__,
                        failure_status,
                    )
                    return {
                        "gene": gene, "pmid": pmid, "title": title,
                        "analysis_type": analysis_type,
                        "status": failure_status, "result": failure_status,
                        "abstract_text": abstract, "paragraphs_text": paragraphs,
                        "reasoning_trace_list": cot.get("reasoning_trace_list", []) if 'cot' in locals() and cot else [],
                        "json_reason": str(e),
                    }
                logger.debug(
                    "Retrying Step 3 relevance task for PMID %s / gene %s / analysis %s "
                    "(attempt %s/%s; %s).",
                    pmid,
                    gene,
                    analysis_type,
                    attempt + 1,
                    MAX_RETRIES,
                    type(e).__name__,
                )
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
    return {"status": "unknown"}

# =================================================================
# === 3. [NEW] Synchronous Batch Logic (Step-by-Step) ===
# =================================================================

async def _wait_for_batch(
    batch_id: str,
    batch_manager: BatchManager,
    step_name: str,
    idle_timeout_seconds: int = BATCH_INACTIVITY_TIMEOUT_SECONDS,
    activity_state_file: Optional[Path] = None,
) -> str:
    """Wait until completion, terminal failure, or 24 hours without observable progress."""
    logger.info("Waiting for Step 3 Batch job %s (%s).", batch_id, step_name)
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
                "Could not persist Step 3 Batch activity state '%s' (%s).",
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
            if status == "completed":
                logger.info("Step 3 Batch job %s completed (%s).", batch_id, step_name)
                return status
            elif status in ["failed", "expired", "cancelled"]:
                logger.warning("Step 3 Batch job %s ended with status '%s' (%s).", batch_id, status, step_name)
                return status
            else:
                idle_seconds = max(0, now - last_activity_at)
                if idle_seconds >= idle_timeout_seconds:
                    logger.warning(
                        "Step 3 Batch job %s (%s) made no progress for %.1f hours; switching to realtime.",
                        batch_id,
                        step_name,
                        idle_seconds / 3600,
                    )
                    return "idle_timeout"
                logger.debug(
                    "Step 3 Batch job %s status is '%s'; idle for %.1f hours (%s).",
                    batch_id,
                    status,
                    idle_seconds / 3600,
                    step_name,
                )
                await asyncio.sleep(min(30, max(1, idle_timeout_seconds - idle_seconds)))
        except Exception as e:
            logger.debug(
                "Could not check Step 3 Batch job %s status (%s); retrying.",
                batch_id,
                type(e).__name__,
            )
            idle_seconds = max(0, time.time() - last_activity_at)
            if idle_seconds >= idle_timeout_seconds:
                logger.warning(
                    "Step 3 Batch status checks made no progress for %.1f hours (%s); switching to realtime.",
                    idle_seconds / 3600,
                    step_name,
                )
                return "idle_timeout"
            await asyncio.sleep(min(30, max(1, idle_timeout_seconds - idle_seconds)))


async def _switch_step_3_to_realtime(
    batch_manager: BatchManager,
    batch_id: str,
    stage_name: str,
    reason: str,
    fallback_marker: Path,
    meta_file: Path,
) -> None:
    if reason != "expired":
        try:
            await asyncio.to_thread(batch_manager.cancel_batch_job, batch_id)
            logger.debug("Requested cancellation of idle Step 3 Batch job %s.", batch_id)
        except Exception as exc:
            logger.warning(
                "Could not cancel Step 3 Batch job %s before realtime fallback (%s).",
                batch_id,
                type(exc).__name__,
            )
    fallback_marker.write_text(
        json.dumps({"batch_id": batch_id, "stage": stage_name, "reason": reason}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if meta_file.exists():
        meta_file.unlink()
    if reason == "idle_timeout":
        message = f"Step 3 Batch stage {stage_name} had no progress for 24 hours; using realtime calls."
    else:
        message = f"Step 3 Batch stage {stage_name} ended with {reason}; using realtime calls."
    raise BatchInactivityTimeout(message)

async def _run_step_by_step_batch(
    unique_tasks,
    analysis_data,
    services,
    analysis_dir,
    batch_run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    batch_manager = services["batch_manager"]
    llm_manager = services["llm_manager"]
    # [关键修复] 1. 统一管理 Batch 文件路径并创建文件夹
    batch_dir = analysis_dir / "batch_files"
    if batch_run_id:
        batch_dir = batch_dir / batch_run_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    state_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model_name": llm_manager.model_name,
                "tasks": unique_tasks,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    state_suffix = state_fingerprint[:12]
    fallback_marker = batch_dir / f"step3_realtime_fallback_{state_suffix}.json"
    if fallback_marker.exists():
        raise BatchInactivityTimeout("This Step 3 task fingerprint is marked for realtime fallback.")

    # 1. 初始化状态文件
    state_file = batch_dir / "step3_cot_state.json"
    task_state = {}
    
    if state_file.exists():
        try:
            with open(state_file, 'r') as f:
                saved_state = json.load(f)
            if saved_state.get("fingerprint") == state_fingerprint:
                task_state = saved_state.get("tasks", {})
        except (OSError, json.JSONDecodeError, AttributeError):
            task_state = {}

    if not task_state:
        for key, task in unique_tasks.items():
            init_msgs = [{"role": "system", "content": "You are a rigorous biomedical research assistant who bases reasoning and answers solely on the information provided by the user, without adding external knowledge or overinterpretation."}]
            bg = task.get('background')
            prefix = f"Article Text:\n'''\n{task.get('input_text')}\n'''"
            if bg: prefix = f"{prefix}\n\n**The research background described by the user:**\n'''\n{bg}\n'''"
            prefix += "\n\nPlease answer the following questions based solely on the content of the provided text:"
            
            task_state[key] = {
                "messages": init_msgs,
                "context_prefix": prefix,
                "analysis_target": task['analysis_target'],
                "background":bg,
                "step_completed": -1
            }
        with open(state_file, 'w') as f:
            json.dump({"fingerprint": state_fingerprint, "tasks": task_state}, f)

    # 2. 循环执行 CoT 4 步
    for step_idx, question_tmpl in enumerate(REASONING_QUESTIONS):
        logger.info("Step 3 Batch is running reasoning substep %s/4.", step_idx + 1)
        tasks_to_process = [
            (key, state)
            for key, state in task_state.items()
            if state['step_completed'] == step_idx - 1
        ]
        
        # A. 检查/提交任务
        meta_file = batch_dir / f"step3_batch_meta_{state_suffix}_s{step_idx}.json"
        batch_id = None
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f: batch_id = json.load(f).get("batch_id")
                if batch_id:
                    status = await asyncio.to_thread(batch_manager.check_batch_status, batch_id)
                    if status == "expired":
                        await _switch_step_3_to_realtime(
                            batch_manager, batch_id, f"reasoning substep {step_idx + 1}/4", "expired", fallback_marker, meta_file
                        )
                    if status in ["failed", "expired", "cancelled"]:
                        logger.warning(
                            "Step 3 Batch subtask %s ended with status '%s'; submitting a replacement.",
                            step_idx + 1,
                            status,
                        )
                        os.remove(meta_file)
                        batch_id = None
            except BatchInactivityTimeout:
                raise
            except Exception as exc:
                logger.warning(
                    "Could not inspect existing Step 3 Batch job %s (%s).",
                    batch_id,
                    type(exc).__name__,
                )
            
        if not batch_id:
            jsonl_path = batch_dir / f"step3_batch_input_s{step_idx}.jsonl"
            count = 0
            logger.debug("Preparing Step 3 Batch input for substep %s/4.", step_idx + 1)
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for key, state in tqdm(
                    tasks_to_process,
                    desc=f"Building Step 3 Batch input for substep {step_idx + 1}/4",
                    disable=not progress_enabled(logger),
                ):
                    # 只处理前一步成功的任务
                    if state['step_completed'] != step_idx - 1: continue
                    
                    target = state['analysis_target']
                    q_text = question_tmpl.format(analysis_target=target)
                    msgs = list(state['messages'])
                    
                    if step_idx == 0:
                        full_content = f"{state['context_prefix']}\n\n**Question 1:**\n{q_text}"
                        msgs.append({"role": "user", "content": full_content})
                    else:
                        msgs.append({"role": "user", "content": f"**Question {step_idx + 1}:**\n{q_text}"})
                        
                    req = {
                        "custom_id": key, "method": "POST", "url": "/v1/chat/completions",
                        "body": {"model": llm_manager.model_name, "messages": msgs, "temperature": 1}
                    }
                    f.write(json.dumps(req, ensure_ascii=False) + "\n")
                    count += 1
            
            if count == 0:
                logger.debug("No Step 3 tasks are pending for substep %s/4.", step_idx + 1)
                continue

            fid = await asyncio.to_thread(batch_manager.upload_file, jsonl_path)
            batch_id = await asyncio.to_thread(batch_manager.create_batch_job, fid, description=f"Step3_S{step_idx + 1}")
            with open(meta_file, 'w', encoding="utf-8") as f:
                json.dump({"batch_id": batch_id, "last_activity_at": time.time()}, f)

        # B. 阻塞等待
        batch_status = await _wait_for_batch(
            batch_id,
            batch_manager,
            f"reasoning substep {step_idx + 1}/4",
            activity_state_file=meta_file,
        )
        if batch_status in {"idle_timeout", "expired"}:
            await _switch_step_3_to_realtime(
                batch_manager,
                batch_id,
                f"reasoning substep {step_idx + 1}/4",
                batch_status,
                fallback_marker,
                meta_file,
            )
        if batch_status != "completed":
            logger.error("Step 3 Batch reasoning substep %s did not complete.", step_idx + 1)
            return []

        # C. 下载并更新
        logger.debug("Downloading results for Step 3 Batch substep %s.", step_idx + 1)
        details = await asyncio.to_thread(batch_manager.get_batch_details, batch_id)
        text = await asyncio.to_thread(batch_manager.download_results, details.output_file_id)

        # [新增代码 START] 将结果保存到本地 JSONL 文件
        result_file_path = batch_dir / f"step3_batch_output_s{step_idx}.jsonl"
        try:
            with open(result_file_path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.debug("Saved raw Step 3 Batch results to '%s'.", result_file_path)
        except Exception as e:
            logger.warning(
                "Could not save the Step 3 Batch output backup '%s' (%s).",
                result_file_path,
                type(e).__name__,
            )
        # [新增代码 END]

        
        updates = 0
        for item in batch_manager.parse_batch_output_to_dict(text):
            if item.get('response', {}).get('status_code') == 200:
                cid = item['custom_id']
                ans = item['response']['body']['choices'][0]['message']['content']
                
                if cid in task_state:
                    st = task_state[cid]
                    tgt = st['analysis_target']
                    q = REASONING_QUESTIONS[step_idx].format(analysis_target=tgt)
                    
                    if step_idx == 0:
                        st['messages'].append({"role": "user", "content": f"{st['context_prefix']}\n\n**Question 1:**\n{q}"})
                    else:
                        st['messages'].append({"role": "user", "content": f"**Question {step_idx + 1}:**\n{q}"})
                        
                    st['messages'].append({"role": "assistant", "content": ans})
                    st['step_completed'] = step_idx
                    updates += 1
                    
        with open(state_file, 'w') as f:
            json.dump({"fingerprint": state_fingerprint, "tasks": task_state}, f)
        logger.info(
            "Step 3 Batch substep %s accepted responses for %s of %s task(s).",
            step_idx + 1,
            updates,
            len(tasks_to_process),
        )
        if updates < len(tasks_to_process):
            logger.warning(
                "Step 3 Batch substep %s did not return usable responses for %s task(s).",
                step_idx + 1,
                len(tasks_to_process) - updates,
            )

    # 3. 最终格式化 Step
    logger.debug("Preparing final Step 3 response formatting.")
    fmt_tasks = [
        (key, state)
        for key, state in task_state.items()
        if state['step_completed'] == 3
    ]
    fmt_meta = batch_dir / f"step3_batch_meta_{state_suffix}_fmt.json"
    fmt_bid = None
    if fmt_meta.exists():
        try:
            with open(fmt_meta, 'r') as f: 
                fmt_bid = json.load(f).get("batch_id")
            
            # 如果找到了 ID，先查一下它的状态
            if fmt_bid:
                status = await asyncio.to_thread(batch_manager.check_batch_status, fmt_bid)
                if status == "expired":
                    await _switch_step_3_to_realtime(
                        batch_manager, fmt_bid, "format", "expired", fallback_marker, fmt_meta
                    )
                if status in ["failed", "expired", "cancelled"]:
                    logger.warning(
                        "Previous Step 3 formatting Batch job %s ended with status '%s'; replacing it.",
                        fmt_bid,
                        status,
                    )
                    # 删除旧文件，并将 ID 置空，触发下方的重新提交逻辑
                    os.remove(fmt_meta)
                    fmt_bid = None
        except BatchInactivityTimeout:
            raise
        except Exception as e:
            logger.warning(
                "Could not inspect existing Step 3 formatting Batch job %s (%s).",
                fmt_bid,
                type(e).__name__,
            )
        
    if not fmt_bid:
        jl_path = batch_dir / "step3_batch_input_fmt.jsonl"
        cnt = 0

        # --- [新增] 增加进度条 ---
        logger.debug("Generating the Step 3 formatting JSONL input.")
        with open(jl_path, 'w', encoding='utf-8') as f:
            for key, state in tqdm(
                fmt_tasks,
                desc="Building Format Batch",
                disable=not progress_enabled(logger),
            ):
                if state['step_completed'] != 3: continue
                
                last_cot_response =state['messages'][-1]['content']
                analysis_target = state['analysis_target']
                bg=state["background"]
                
                format_desc = f"""The output in JSON format should contain two fields: "result" and "reason". If the text includes statements directly related to "{analysis_target}", set "result" to "yes"; otherwise, set it to "no". Provide a detailed explanation in the "reason" field."""
                
                if bg: format_desc = f"**The research background described by the user:**\n'''\n{bg}\n'''\n\n{format_desc}"
                msgs = [
                    {"role": "system", "content": "You are a formatting function. You will only output the content in the transformed format and nothing else."},
                    {"role": "user", "content": f"**Content**:\n{last_cot_response}\n**Formatting Requirements**:\n{format_desc}"}
                ]
                
                req = {
                    "custom_id": key, "method": "POST", "url": "/v1/chat/completions",
                    "body": {
                        "model": llm_manager.model_name, 
                        "messages": msgs,
                        "temperature": 1, "response_format": {"type": "json_object"}
                    }
                }
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
                cnt += 1
        
        if cnt > 0:
            fid = await asyncio.to_thread(batch_manager.upload_file, jl_path)
            fmt_bid = await asyncio.to_thread(batch_manager.create_batch_job, fid, description="Step3_Fmt")
            with open(fmt_meta, 'w', encoding="utf-8") as f:
                json.dump({"batch_id": fmt_bid, "last_activity_at": time.time()}, f)
        else: return []

    fmt_status = await _wait_for_batch(
        fmt_bid, batch_manager, "Format", activity_state_file=fmt_meta
    )
    if fmt_status in {"idle_timeout", "expired"}:
        await _switch_step_3_to_realtime(
            batch_manager, fmt_bid, "format", fmt_status, fallback_marker, fmt_meta
        )
    if fmt_status != "completed": return []

    # 4. 收集最终结果
    det = await asyncio.to_thread(batch_manager.get_batch_details, fmt_bid)
    txt = await asyncio.to_thread(batch_manager.download_results, det.output_file_id)

    # [新增代码 START] 将结果保存到本地 JSONL 文件
    result_file_path = batch_dir / f"step3_batch_output_fmt.jsonl"
    try:
        with open(result_file_path, "w", encoding="utf-8") as f:
            f.write(txt)
        logger.debug("Saved raw Step 3 Batch results to '%s'.", result_file_path)
    except Exception as e:
        logger.warning(
            "Could not save the Step 3 formatting output backup '%s' (%s).",
            result_file_path,
            type(e).__name__,
        )
        # [新增代码 END]

    
    final_res = []
    batch_items = batch_manager.parse_batch_output_to_dict(txt)
    for item in batch_items:
        if item.get('response', {}).get('status_code') == 200:
            cid = item['custom_id']
            cnt = item['response']['body']['choices'][0]['message']['content']
            parsed = _extract_and_parse_json(cnt)
            
            if parsed and cid in task_state:
                pmid, gene, atype = cid.split('|', 2)
                
                
                # 重建标准 Trace List [Q1, A1, Q2, A2...]
                full_trace = []
                answers = [m['content'] for m in task_state[cid]['messages'] if m['role'] == 'assistant']
                for i, q_tmpl in enumerate(REASONING_QUESTIONS):
                    target = task_state[cid]['analysis_target']
                    full_trace.append(q_tmpl.format(analysis_target=target)) # Q
                    if i < len(answers): full_trace.append(answers[i])       # A
                
                result = _normalize_relevance_result(parsed.get("result"))
                status = "success" if result in {"yes", "no"} else "parse_failed"
                final_res.append({
                    "gene": gene, "pmid": pmid, "analysis_type": atype, "status": status,
                    "reasoning_trace_list": full_trace,
                    "result": result, "json_reason": str(parsed.get("reason", "")),
                })
            elif cid in task_state:
                pmid, gene, atype = cid.split('|', 2)
                final_res.append({
                    "gene": gene, "pmid": pmid, "analysis_type": atype,
                    "status": "parse_failed", "result": "parse_failed",
                    "reasoning_trace_list": [], "json_reason": "Batch formatting response was missing or invalid JSON.",
                })
    logger.info(
        "Step 3 formatting returned %s usable result(s) for %s expected task(s).",
        len(final_res),
        len(fmt_tasks),
    )
    if len(final_res) < len(fmt_tasks):
        logger.warning("Step 3 formatting omitted %s expected task result(s).", len(fmt_tasks) - len(final_res))
    return final_res


async def _run_realtime_relevance_tasks(unique_tasks, llm_manager) -> List[Dict[str, Any]]:
    """Assess candidate evidence in realtime and show active gene/article tasks."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHECKS)
    task_list = list(unique_tasks.values())
    active_tasks = []
    relationship_labels = {
        "gene_vs_function": "G–F",
        "gene_vs_context": "G–C",
        "function_vs_context": "F–C",
    }

    with tqdm(
        total=len(task_list),
        desc="Step 3 evidence",
        unit="assessment",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}",
        disable=not progress_enabled(logger),
    ) as progress:
        def refresh_active_tasks() -> None:
            progress.set_postfix_str(active_tasks[-1] if active_tasks else "")

        async def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
            label = (
                f"{task['gene']} | PMID {task['pmid']} | "
                f"{relationship_labels.get(task['agent_type'], task['agent_type'])}"
            )
            started = False

            def mark_started() -> None:
                nonlocal started
                started = True
                active_tasks.append(label)
                refresh_active_tasks()

            try:
                return await _process_single_agent_task(
                    llm_manager,
                    semaphore,
                    task['abstract_text'],
                    task['paragraphs_text'],
                    task['background'],
                    task['analysis_target'],
                    task['pmid'],
                    task['gene'],
                    task['title'],
                    task['agent_type'],
                    on_started=mark_started,
                )
            finally:
                if started:
                    active_tasks.remove(label)
                    progress.update(1)
                    refresh_active_tasks()

        tasks = [run_task(task) for task in task_list]
        return [await future for future in asyncio.as_completed(tasks)]

# =================================================================
# === 4. Main Entry Point ===
# =================================================================

async def run_relevance_filter_step(
    analysis_data: AnalysisData,
    services: Dict[str, Any],
    minimum_cosine_similarity: float = 0.5,
    batch_run_id: Optional[str] = None,
) -> AnalysisData:

    analysis_dir = Path(analysis_data.project_folder) / f"analysis_{analysis_directory_name(analysis_data.function_name)}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    db_path = analysis_dir / "step2_embeddings.db"

    def persist_stage_status(status: str, details: Optional[Dict[str, Any]] = None) -> None:
        status_writer = MilvusManager(
            str(db_path), "biomed_sentences",
            services["embedding_manager"].dimensions,
        )
        try:
            status_writer.save_pipeline_stage_status("step3", status, details)
        finally:
            status_writer.close()
        analysis_data.relevance_filter_status = status

    # 1. 准备数据
    recalled_df = analysis_data.recalled_sentences_df
    if recalled_df.empty:
        persist_stage_status("no_candidates", {"reason": "empty_recall_set"})
        logger.warning("Step 3 Assessment of Candidate Evidence skipped because sentence-level semantic retrieval produced no candidate evidence.")
        analysis_data.relevance_filtered_df = pd.DataFrame()
        return analysis_data
    filtered_df = recalled_df[
        recalled_df['cosine_similarity'] >= minimum_cosine_similarity
    ].copy()
    if filtered_df.empty:
        persist_stage_status(
            "no_candidates",
            {"reason": "below_recall_threshold", "threshold": minimum_cosine_similarity},
        )
        logger.warning(
            "Step 3 Assessment of Candidate Evidence found no candidate sentences above cosine similarity %.4f.",
            minimum_cosine_similarity,
        )
        analysis_data.relevance_filtered_df = pd.DataFrame()
        return analysis_data
    
    # 任务去重与准备
    unique_pairs = filtered_df[['PMID', 'Gene']].drop_duplicates().to_dict('records')
    unique_tasks = {}
    
    for p in unique_pairs:
        pmid, gene = p['PMID'], p['Gene']
        ctx = _get_input_text_for_pmid(pmid, gene, analysis_data, analysis_dir, filtered_df)
        if not ctx: continue
        
        parts = []
        if ctx.get('title'): parts.append(f"# Article Title:{ctx['title']}")
        if ctx.get('abstract_text'): parts.append(f"# Article Abstract:\n{ctx['abstract_text']}")
        if ctx.get('paragraphs_text'): parts.append(f"# Highly Relevant Sentences:\n{ctx['paragraphs_text']}")
        input_txt = "\n\n---\n\n".join(parts)
        
        for atype in ["gene_vs_function", "gene_vs_context", "function_vs_context"]:
            prep = _prepare_analysis_target(
                analysis_data.research_background_template,
                gene,
                analysis_data.function_name,
                atype,
            )
            key = f"{pmid}|{gene}|{atype}"
            unique_tasks[key] = {
                "pmid": pmid, "gene": gene, "agent_type": atype,
                "input_text": input_txt,
                "analysis_target": prep['context'],
                "background": prep['background'],
                "title": ctx.get('title', ''),
                "abstract_text": ctx.get('abstract_text', ''),
                "paragraphs_text": ctx.get('paragraphs_text', '')
            }

    candidate_pmids_by_gene = defaultdict(set)
    for pair in unique_pairs:
        candidate_pmids_by_gene[str(pair["Gene"])].add(str(pair["PMID"]))
    candidate_breakdown = "; ".join(
        f"{gene}: {len(candidate_pmids_by_gene.get(gene, set()))} PMID(s)"
        for gene in analysis_data.genes
    )
    # 2. 判断模式
    llm_manager = services["llm_manager"]
    use_batch = BATCH_API_ENABLED and llm_manager.model_name in BATCH_SUPPORTED_MODELS
    
    all_results = []
    missing_text_pairs = 0
    missing_text_labels = []
    for pair in unique_pairs:
        pmid, gene = str(pair['PMID']), str(pair['Gene'])
        if any(task['pmid'] == pmid and task['gene'] == gene for task in unique_tasks.values()):
            continue
        missing_text_pairs += 1
        missing_text_labels.append(f"{gene} / PMID {pmid}")
        for atype in ["gene_vs_function", "gene_vs_context", "function_vs_context"]:
            all_results.append({
                "gene": gene, "pmid": pmid, "analysis_type": atype,
                "status": "failed", "result": "failed",
                "reasoning_trace_list": [],
                "json_reason": "No source text was available for this PMID/gene pair.",
            })
    if missing_text_pairs:
        logger.warning(
            "Step 3 could not assess candidate evidence for %s gene-PMID pair(s) because no source text was available: %s.",
            missing_text_pairs,
            "; ".join(missing_text_labels),
        )

    execution_mode = "Batch API" if use_batch else "real-time model calls"
    logger.info(
        "Step 3 Assessment of Candidate Evidence is assessing three relationships (G–F, G–C, F–C) for %s gene-PMID pair(s) across %s target gene(s) (%s), using %s.",
        len(unique_pairs),
        len(analysis_data.genes),
        candidate_breakdown,
        execution_mode,
    )
    
    if use_batch:
        try:
            unique_results = await _run_step_by_step_batch(
                unique_tasks,
                analysis_data,
                services,
                analysis_dir,
                batch_run_id=batch_run_id,
            )
        except BatchInactivityTimeout as exc:
            logger.warning("%s Switching all Step 3 tasks to realtime mode.", exc)
            unique_results = await _run_realtime_relevance_tasks(unique_tasks, llm_manager)
        
        # 结果补全与扩展
        task_map_simple = {f"{t['pmid']}|{t['gene']}|{t['agent_type']}": t for t in unique_tasks.values()}
        
        for res in unique_results:
            key = f"{res['pmid']}|{res['gene']}|{res['analysis_type']}"
            if key in task_map_simple:
                orig = task_map_simple[key]
                res['title'] = orig['title']
                res['abstract_text'] = orig['abstract_text']
                res['paragraphs_text'] = orig['paragraphs_text']
            all_results.append(res)

        returned_keys = {
            f"{res.get('pmid')}|{res.get('gene')}|{res.get('analysis_type')}"
            for res in unique_results
        }
        for key, task in unique_tasks.items():
            if key not in returned_keys:
                all_results.append({
                    "gene": task['gene'], "pmid": task['pmid'], "title": task['title'],
                    "analysis_type": task['agent_type'], "status": "failed", "result": "failed",
                    "abstract_text": task['abstract_text'], "paragraphs_text": task['paragraphs_text'],
                    "reasoning_trace_list": [],
                    "json_reason": "Batch processing did not return a result for this task.",
                })
            
    else:
        # 实时模式 (Real-time)
        all_results.extend(await _run_realtime_relevance_tasks(unique_tasks, llm_manager))

    # 3. 保存与返回 (复用原有逻辑)
    if not all_results: return analysis_data
    
    initial_df = pd.DataFrame(all_results)
    
    # Save to DB
    df_db = initial_df.copy()
    df_db['model_name'] = llm_manager.model_name
    df_db['record_id'] = df_db.apply(
        lambda result_row: f"{result_row['pmid']}_{result_row['gene']}_{result_row['analysis_type']}_{result_row['model_name']}",
        axis=1,
    )
    df_db['reasoning_trace'] = df_db['reasoning_trace_list'].apply(
        lambda trace_items: json.dumps(trace_items)
    )
    df_db.rename(columns={'json_reason': 'json_reason'}, inplace=True)
    
    mm = MilvusManager(
        str(db_path), "biomed_sentences",
        services["embedding_manager"].dimensions,
    )
    try:
        recs = df_db[['record_id', 'pmid', 'gene', 'analysis_type', 'model_name', 'result', 'json_reason', 'abstract_text', 'paragraphs_text', 'reasoning_trace']].to_dict('records')
        mm.upsert_llm_analysis_batch(recs)
        persisted_by_id = {
            row["record_id"]: row for row in mm.load_llm_analysis_results()
            if row.get("record_id") in set(df_db["record_id"])
        }
        if len(persisted_by_id) != len(recs):
            raise RuntimeError(
                f"Step 3 persisted {len(persisted_by_id)} of {len(recs)} result records."
            )
    finally:
        mm.close()

    # Rehydrate the current run from Milvus so downstream steps consume the
    # canonical, lossless values rather than the pre-persistence DataFrame.
    title_by_pair = {
        (str(row.get("PMID", "")), str(row.get("Gene", ""))): row.get("title", "")
        for _, row in analysis_data.raw_papers_df.iterrows()
    }
    persisted_rows = []
    for record in recs:
        stored = persisted_by_id.get(record["record_id"])
        if not stored:
            continue
        try:
            trace = json.loads(stored.get("reasoning_trace", "[]"))
        except (TypeError, json.JSONDecodeError):
            trace = []
        result = _normalize_relevance_result(stored.get("result"))
        persisted_rows.append({
            "gene": stored["gene"], "pmid": stored["pmid"],
            "title": title_by_pair.get((stored["pmid"], stored["gene"]), ""),
            "analysis_type": stored["analysis_type"],
            "status": "success" if result in {"yes", "no"} else result,
            "result": result, "json_reason": stored.get("json_reason", ""),
            "abstract_text": stored.get("abstract_text", ""),
            "paragraphs_text": stored.get("paragraphs_text", ""),
            "reasoning_trace_list": trace,
        })
    initial_df = pd.DataFrame(persisted_rows)

    # Save CSV
    final_recs = []
    for _, row in initial_df.iterrows():
        r = row.to_dict()
        trace = r.pop('reasoning_trace_list', [])
        for i in range(4):
            r[f'Q{i+1}'] = trace[i*2] if len(trace)>i*2 else None
            r[f'A{i+1}'] = trace[i*2+1] if len(trace)>i*2+1 else None
        final_recs.append(r)
        
    df_csv = pd.DataFrame(final_recs)
    for c in ['json_reason', 'abstract_text', 'Q1','A1','Q2','A2','Q3','A3','Q4','A4']:
        if c in df_csv.columns: df_csv[c] = df_csv[c].fillna('').apply(_clean_csv_text)
    
    cols = ["gene", "pmid", "title", "analysis_type", "result", "abstract_text", "Q1", "A1", "Q2", "A2", "Q3", "A3", "Q4", "A4", "json_reason", "status"]
    saver = AnalysisSaver(analysis_dir)
    saver.save_relevance_results(df_csv[[c for c in cols if c in df_csv.columns]])
    
    analysis_data.relevance_filtered_df = initial_df
    status_counts = initial_df["status"].value_counts().to_dict()
    valid_count = int(status_counts.get("success", 0))
    if valid_count == len(initial_df):
        stage_status = "completed"
    elif valid_count:
        stage_status = "partial"
    else:
        stage_status = "failed"
    persist_stage_status(stage_status, {
        "total": int(len(initial_df)),
        "success": valid_count,
        "parse_failed": int(status_counts.get("parse_failed", 0)),
        "failed": int(sum(count for status, count in status_counts.items() if status not in {"success", "parse_failed"})),
    })
    relevant_count = int(
        initial_df["result"].fillna("").astype(str).str.strip().str.casefold().eq("yes").sum()
    )
    per_gene_outcomes = []
    for gene in analysis_data.genes:
        gene_rows = initial_df[initial_df["gene"].astype(str).str.casefold() == gene.casefold()]
        gene_supported = int(
            gene_rows["result"].fillna("").astype(str).str.strip().str.casefold().eq("yes").sum()
        )
        per_gene_outcomes.append(f"{gene}: {gene_supported}/{len(gene_rows)} supported")
    outcome_breakdown = ", ".join(
        f"{status}={count}" for status, count in sorted(status_counts.items())
    )
    logger.info(
        "Step 3 Assessment of Candidate Evidence finished with status '%s': %s of %s relationship assessment(s) were supported; by target gene: %s; response status counts: %s.",
        stage_status,
        relevant_count,
        len(initial_df),
        "; ".join(per_gene_outcomes),
        outcome_breakdown,
    )
    return analysis_data
