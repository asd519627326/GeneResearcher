# src/connectors/batch_manager.py

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)
BATCH_INACTIVITY_TIMEOUT_SECONDS = 24 * 60 * 60

class BatchManager:
    """
    OpenAI 兼容的 Batch API 管理器。
    """
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        if not api_key:
            raise ValueError("api_key is required for BatchManager.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.debug("BatchManager initialized.")

    def upload_file(self, file_path: Path) -> str:
        """上传 JSONL 文件用于 Batch 处理。"""
        if not file_path.exists():
            raise FileNotFoundError(f"Batch file not found: {file_path}")
        logger.debug("Uploading Batch input file '%s'.", file_path)
        try:
            with open(file_path, "rb") as f:
                file_object = self.client.files.create(file=f, purpose="batch")
        except Exception as exc:
            logger.error(
                "Batch input upload failed for '%s' (%s).",
                file_path,
                type(exc).__name__,
            )
            raise
        logger.debug("Uploaded Batch input file as %s.", file_object.id)
        return file_object.id

    def create_batch_job(self, input_file_id: str, endpoint: str = "/v1/chat/completions", description: str = "Batch Job") -> str:
        """创建 Batch 任务。"""
        logger.debug("Creating a Batch job from input file %s.", input_file_id)
        try:
            batch = self.client.batches.create(
                input_file_id=input_file_id,
                endpoint=endpoint,
                completion_window="24h",
                metadata={"ds_description": description}
            )
        except Exception as exc:
            logger.error(
                "Batch job creation failed for input file %s (%s).",
                input_file_id,
                type(exc).__name__,
            )
            raise
        logger.debug("Created Batch job %s.", batch.id)
        return batch.id

    def check_batch_status(self, batch_id: str) -> str:
        """查询任务状态。"""
        batch = self.client.batches.retrieve(batch_id)
        return batch.status

    def check_batch_activity(self, batch_id: str):
        """Return status plus observable request-count progress for idle detection."""
        batch = self.client.batches.retrieve(batch_id)
        counts = getattr(batch, "request_counts", None)
        activity = None
        if counts is not None:
            activity = (
                getattr(counts, "total", None),
                getattr(counts, "completed", None),
                getattr(counts, "failed", None),
            )
        return batch.status, activity

    def cancel_batch_job(self, batch_id: str):
        """Request cancellation before switching an idle Batch to realtime calls."""
        return self.client.batches.cancel(batch_id)

    # --- [关键修复] 补充缺失的方法 ---
    def get_batch_details(self, batch_id: str):
        """获取完整的 Batch 对象信息（包含 output_file_id）。"""
        return self.client.batches.retrieve(batch_id)

    def download_results(self, output_file_id: str) -> str:
        """下载结果文件内容。"""
        logger.debug("Downloading Batch output file %s.", output_file_id)
        try:
            content = self.client.files.content(output_file_id)
            return content.text
        except Exception as exc:
            logger.error(
                "Batch output download failed for file %s (%s).",
                output_file_id,
                type(exc).__name__,
            )
            raise

    def parse_batch_output_to_dict(self, jsonl_content: str) -> List[Dict[str, Any]]:
        """解析 JSONL 结果。"""
        results = []
        malformed_count = 0
        first_malformed_line = None
        for line_number, line in enumerate(jsonl_content.splitlines(), start=1):
            if not line.strip(): continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                malformed_count += 1
                if first_malformed_line is None:
                    first_malformed_line = line_number
        if malformed_count:
            logger.warning(
                "Ignored %s malformed Batch output line(s); the first was line %s.",
                malformed_count,
                first_malformed_line,
            )
        return results
