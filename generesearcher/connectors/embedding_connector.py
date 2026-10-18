# src/connectors/embedding_connector.py

import logging
import os
import time
from typing import List, Any, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIError
logger = logging.getLogger(__name__)


# 在这里集中管理所有支持的远程模型和它们的配置
REMOTE_EMBEDDING_CONFIGS = {
    "gitee": {
        "endpoint": "https://ai.gitee.com/v1",
        "default_model": "Qwen3-Embedding-0.6B",
        "api_key_env": "EMBEDDING_API_KEY",
        "default_headers": {"X-Failover-Enabled": "true"}, # 新增 headers
        "dimension": 1024 
    },
    # --- 新增: DashScope (通义千问) 配置 ---
    "dashscope": {
        "default_model": "text-embedding-v3",
        "api_key_env": "EMBEDDING_API_KEY",
        "dimension": 1024
        # DashScope SDK 不需要 endpoint 或 headers
    },
    # --- 未来可以轻松扩展 ---
}



class EmbeddingManager:
    """
    A unified manager for obtaining vectors from remote embedding APIs.
    """
    def __init__(
        self,
        provider: str,
        model_name: Optional[str] = None,
    ):
        self.provider = provider.lower() if provider else "gitee"
        self.model_name = model_name
        self.remote_client: Optional[OpenAI] = None
        self.dimensions: Optional[int] = None

        if self.provider not in REMOTE_EMBEDDING_CONFIGS:
            raise ValueError(
                f"Unsupported remote embedding provider: '{self.provider}'. "
                f"Supported providers are {list(REMOTE_EMBEDDING_CONFIGS)}."
            )

        provider_config = REMOTE_EMBEDDING_CONFIGS[self.provider]
        if not self.model_name:
            self.model_name = provider_config["default_model"]
            logger.debug(
                "Using default embedding model '%s' for provider '%s'.",
                self.model_name,
                self.provider,
            )
        self._initialize_remote_client(provider_config)

        logger.debug(
            "Initialized remote embedding with provider '%s' and model '%s'.",
            self.provider,
            self.model_name,
        )

    # --- 封装查询格式化逻辑 ---
    def format_query(self, query: str, task_description: str) -> str:
        """根据提供商的要求格式化原始查询字符串。"""
        if self.provider == "gitee":
            return f"Instruct: {task_description}\nQuery: {query}"
        elif self.provider == 'dashscope':
            return query
        else:
            return query


    
    # --- 核心修改: 初始化 OpenAI 客户端 ---
    def _initialize_remote_client(self, config: dict):
        """准备远程API客户端或设置SDK。"""
        api_key_env = config['api_key_env']
        api_key = os.getenv(api_key_env)
        self.dimensions = config.get('dimension')
        if not self.dimensions:
            raise ValueError(f"The 'dimension' of the embedding must be specified in the configuration for the provider '{self.provider}'.")
        if not api_key:
            raise ValueError(f"API key environment variable '{api_key_env}' is required for provider '{self.provider}' but not set.")

        try:
            # --- 分发器逻辑 ---
            if self.provider == 'gitee':
                self.remote_client = OpenAI(
                    base_url=config['endpoint'],
                    api_key=api_key,
                    default_headers=config.get('default_headers'),
                    timeout=20.0,
                    max_retries=2,
                )
                logger.debug(
                    "Configured embedding client for provider '%s' (timeout=%ss, retries=%s).",
                    self.provider,
                    self.remote_client.timeout,
                    self.remote_client.max_retries,
                )
            elif self.provider == 'dashscope':
                # DashScope SDK 在使用时才导入，避免成为硬依赖
                import dashscope
                dashscope.api_key = api_key
                self.remote_client = True  # 表示初始化成功
                logger.debug("Configured the DashScope embedding client.")
            
        except ImportError as exc:
            raise ModuleNotFoundError(
                "DashScope embedding requires the optional dependency. "
                "Install GeneResearcher with the 'dashscope' extra."
            ) from exc
        except Exception as e:
            raise RuntimeError(
                f"Could not initialize the embedding client for provider '{self.provider}'."
            ) from e

    def get_embeddings(self, sentences: List[str], text_type: str = 'document') -> Optional[np.ndarray]:
        """
        获取句子列表的向量。
        新增 text_type 参数以支持 DashScope 等模型。
        """
        if not sentences:
            return np.empty((0, self.dimensions), dtype=np.float32)
        blank_indices = [
            index for index, sentence in enumerate(sentences)
            if not isinstance(sentence, str) or not sentence.strip()
        ]
        if blank_indices:
            raise ValueError(
                f"Embedding input contains {len(blank_indices)} blank or non-text item(s); "
                "every input must produce a corresponding vector."
            )

        return self._get_embeddings_remote(sentences, text_type)

    def _get_embeddings_remote(self, sentences: List[str], text_type: str) -> Optional[np.ndarray]:
        """
        根据提供商分发到具体的远程 embedding 获取函数。
        """
        if not self.remote_client:
            raise RuntimeError("Remote embedding client is not initialized.")
        
        if self.provider == 'gitee':
            return self._get_embeddings_openai(sentences)
        elif self.provider == 'dashscope':
            return self._get_embeddings_dashscope(sentences, text_type)

        raise RuntimeError(f"No remote embedding implementation for provider: {self.provider}")

    def _get_embeddings_openai(self, sentences: List[str]) -> Optional[np.ndarray]:
        """
        使用 OpenAI 客户端并发获取向量，只有完整成功时才返回结果。
        """
        indexed_sentences = list(enumerate(sentences))

        def fetch_single_embedding(sentence_tuple: tuple) -> tuple:
            original_index, sentence = sentence_tuple
            try:
                response = self.remote_client.embeddings.create(
                    input=sentence,
                    model=self.model_name,
                    dimensions=self.dimensions,
                )
                if not response.data or not response.data[0].embedding:
                    return original_index, None, "API response contained no embedding"
                vector = np.array(response.data[0].embedding, dtype=np.float32)
                if vector.shape != (self.dimensions,):
                    return original_index, None, f"unexpected vector shape {vector.shape}"
                return original_index, vector, None
            except APIError as e:
                return original_index, None, str(e)
            except Exception as e:
                return original_index, None, str(e)

        max_workers = 16
        results_by_index = {}
        failures = []
        logger.debug(
            "Sending %s sentences to Gitee API using %s concurrent workers.",
            len(indexed_sentences),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sentence = {
                executor.submit(fetch_single_embedding, item): item
                for item in indexed_sentences
            }
            for future in as_completed(future_to_sentence):
                try:
                    index, vector, error = future.result()
                    if vector is None:
                        failures.append((index, error or "unknown embedding error"))
                    else:
                        results_by_index[index] = vector
                except Exception as exc:
                    sentence_tuple = future_to_sentence[future]
                    failures.append((sentence_tuple[0], str(exc)))

        if failures or len(results_by_index) != len(sentences):
            failed_indices = sorted(index for index, _ in failures)
            raise RuntimeError(
                f"Gitee embedding failed for {len(failures)} of {len(sentences)} inputs "
                f"(indices: {failed_indices[:20]}). No partial vectors were returned."
            )

        return np.vstack([results_by_index[index] for index in range(len(sentences))])

    def _get_embeddings_dashscope(self, sentences: List[str], text_type: str) -> Optional[np.ndarray]:
        """
        使用 DashScope SDK 和批处理为句子列表获取向量。
        """
        try:
            import dashscope
            from http import HTTPStatus
        except ImportError as e:
            raise ModuleNotFoundError(
                "DashScope provider requires 'dashscope'. Install it with `pip install dashscope`."
            ) from e

        all_embeddings = []
        batch_size = 10  # text-embedding-v3/v4 的最大批次大小
        max_attempts = 3
        
        logger.debug(
            "Requesting %s embedding(s) from DashScope in batches of %s.",
            len(sentences),
            batch_size,
        )

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            last_error = None
            for attempt in range(max_attempts):
                try:
                    resp = dashscope.TextEmbedding.call(
                        model=self.model_name,
                        input=batch,
                        text_type=text_type,
                        dimension=self.dimensions,
                        request_timeout=20,
                    )
    
                    if resp.status_code == HTTPStatus.OK:
                        # DashScope 返回的 embedding 顺序与输入 batch 顺序一致
                        batch_embeddings = [np.array(e['embedding'], dtype=np.float32) for e in resp.output['embeddings']]
                        if len(batch_embeddings) != len(batch) or any(
                            vector.shape != (self.dimensions,) for vector in batch_embeddings
                        ):
                            raise RuntimeError(
                                f"DashScope returned {len(batch_embeddings)} vectors for "
                                f"a batch of {len(batch)} inputs, or a vector had an unexpected dimension."
                            )
                        all_embeddings.extend(batch_embeddings)
                        break
                    else:
                        last_error = RuntimeError(
                            f"DashScope API error {resp.code}: {resp.message}"
                        )
                        logger.debug(
                            "DashScope API error on batch starting at index %s "
                            "(attempt %s/%s; code=%s).",
                            i,
                            attempt + 1,
                            max_attempts,
                            getattr(resp, "code", "unknown"),
                        )
                except Exception as e:
                    last_error = e
                    logger.debug(
                        "DashScope exception on batch starting at index %s "
                        "(attempt %s/%s; %s).",
                        i,
                        attempt + 1,
                        max_attempts,
                        type(e).__name__,
                    )
                if attempt < max_attempts - 1:
                    backoff_time = min(2 ** attempt, 4)
                    logger.debug("Retrying DashScope batch in %.2f seconds.", backoff_time)
                    time.sleep(backoff_time)
                else:
                    raise RuntimeError(
                        f"DashScope embedding failed for batch starting at index {i} "
                        f"after {max_attempts} attempts; no partial vectors were returned."
                    ) from last_error

        if len(all_embeddings) != len(sentences):
            raise RuntimeError(
                f"DashScope returned {len(all_embeddings)} vectors for {len(sentences)} inputs."
            )
        return np.vstack(all_embeddings)
    
