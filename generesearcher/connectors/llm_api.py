# src/connectors/llm_api.py (VERSIÓN MODIFICADA)

import time
import re
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from typing import List, Dict, Optional,Union,Any
import logging

logger = logging.getLogger(__name__)




CACHE_SUPPORTED_MODELS = {
    "qwen-max", "qwen-max-latest", "qwen-plus", "qwen-turbo", "qwen-flash",
    "qwen3-max", "qwen3-coder-plus", "qwen3-coder-flash" 
}

# --- 配置 ---
BATCH_SUPPORTED_MODELS = {
    "qwen3-max", "qwen-max", "qwen-max-latest",
    "qwen-plus", "qwen-plus-latest",
    "qwen-flash", "qwen-turbo", "qwen-turbo-latest", "qwen-long", "qwen-long-latest",
    "qwq-plus", "qwq-32b-preview", "deepseek-r1", "deepseek-v3"
}

# Temporarily keep analysis stages on real-time calls until the Batch path is tested.
BATCH_API_ENABLED = False

class LLMManager:
    """
    Manages the configuration and calls to a large language model (LLM) through an interface compatible with OpenAI.
    """
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        """
        Initializes the LLMManager with the necessary configuration.
        Args: 
            model_name (str): The name of the model to use (e.g., 'qwen-max').            
            api_key (str): The API key for the service.            
            base_url (Optional[str]): The API endpoint (if different from the default OpenAI one).
        """
        if not model_name or not api_key:
            raise ValueError("model_name and api_key are required to initialize LLMManager.")
            
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            logger.debug("Initialized the LLM client for model '%s'.", self.model_name)
        except Exception as e:
            raise RuntimeError("Could not initialize the configured LLM client.") from e


    def create_cached_content(self, text: str, min_len: int = 1024) -> Union[str, List[Dict[str, Any]]]:
        """
        如果当前模型支持缓存且文本长度满足要求，则返回带 cache_control 的结构；
        否则返回原始文本字符串。
        
        Args:
            text: 要发送的文本内容
            min_len: 触发缓存的最小字符长度估计值 (Token数通常比字符数少，建议保守一点，比如 1500 chars)
        """
        # 1. 检查模型是否支持 (模糊匹配，只要包含关键词即可)
        is_supported = any(m in self.model_name for m in CACHE_SUPPORTED_MODELS)
        
        # 2. 检查长度 (显式缓存至少需要 1024 Token)
        # 简单的字符长度估算：1 Token ≈ 1.5 ~ 2 字符 (英文) 或 1 ~ 1.5 字符 (中文)
        # 设定阈值为 1200 字符，低于此值不尝试缓存，避免创建无效缓存或浪费
        if is_supported and len(text) > min_len:
            logger.debug(
                "Enabling prompt caching for %s input characters with model '%s'.",
                len(text),
                self.model_name,
            )
            return [
                {
                    "type": "text",
                    "text": text,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        
        return text


    
    def call(self,
             # 注意：messages_list 的类型注解需要放宽，因为 content 可能是 list 了
             messages_list: List[Dict[str, Union[str, List[Dict]]]],
             max_retries: int = 5,
             retry_delay: int = 5,
             temperature: float = 1,
             response_format: Optional[dict] = None,
             request_context: Optional[str] = None,
             ) -> Optional[ChatCompletion]:
        """
        Make a call to the configured LLM with retry logic.
        """
        # Keep request content out of logs; biomedical text may be sensitive.
        if logger.isEnabledFor(logging.DEBUG):
            last_user_msg = None
            for msg in reversed(messages_list):
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    last_user_msg = msg.get('content', '')
                    break
            if last_user_msg:
                context_label = (
                    f" Context={request_context}"
                    if request_context
                    else ""
                )
                logger.debug(
                    "LLM request started.%s model=%s temperature=%s messages=%s user_content_chars=%s",
                    context_label,
                    self.model_name,
                    temperature,
                    len(messages_list),
                    len(last_user_msg) if isinstance(last_user_msg, str) else len(str(last_user_msg)),
                )

        create_args = {
            "model": self.model_name,
            "messages": messages_list,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        if response_format:
            create_args["response_format"] = response_format

        request_started_at = time.monotonic()
        for attempt in range(max_retries):
            try:
                stream = self.client.chat.completions.create(**create_args, timeout=180.0)
                
                full_content_parts, current_role, finish_reason, usage_info, system_fingerprint = [], "assistant", None, None, None
                last_chunk = None
                
                for chunk in stream:
                    last_chunk = chunk
                    if hasattr(chunk, 'system_fingerprint') and chunk.system_fingerprint:
                        system_fingerprint = chunk.system_fingerprint
                    if chunk.usage:
                        usage_info = chunk.usage
                    if not chunk.choices: continue
                    
                    choice = chunk.choices[0]
                    if choice.delta:
                        if choice.delta.role: current_role = choice.delta.role
                        if choice.delta.content is not None: full_content_parts.append(choice.delta.content)
                    if choice.finish_reason: finish_reason = choice.finish_reason
                
                full_content = "".join(full_content_parts)

                # Reconstruct response
                reconstructed_message = ChatCompletionMessage(content=full_content, role=current_role)
                reconstructed_choice = Choice(finish_reason=finish_reason if finish_reason else "stop", index=0, message=reconstructed_message)
                last_chunk_id = last_chunk.id if last_chunk else f"chatcmpl-stream-{time.time_ns()}"
                
                reconstructed_completion = ChatCompletion(
                    id=last_chunk_id, choices=[reconstructed_choice], created=int(time.time()),
                    model=self.model_name, object="chat.completion", system_fingerprint=system_fingerprint, usage=usage_info
                )
                logger.debug(
                    "LLM request completed with model=%s after %.1f seconds; output_chars=%s.",
                    self.model_name,
                    time.monotonic() - request_started_at,
                    len(full_content),
                )
                return reconstructed_completion

            except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
                logger.debug(
                    "LLM request failed on attempt %s/%s (%s).",
                    attempt + 1,
                    max_retries,
                    type(e).__name__,
                )
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error("LLM request failed after %s attempts.", max_retries)
                    return None
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(
                        "Unexpected LLM request error on attempt %s/%s (%s); retrying.",
                        attempt + 1,
                        max_retries,
                        type(e).__name__,
                    )
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.exception("LLM request failed after %s attempts.", max_retries)
                    return None
        return None

def get_llm_response_content(response: Optional[ChatCompletion]) -> Optional[str]:
    """Extract the text content from a ChatCompletion response."""
    if response is None:
        return None
    try:
        content = response.choices[0].message.content
        try:
            # Usage metadata is optional for providers that do not report cache tokens.
            cache_input_token = response.usage.prompt_tokens_details.cache_creation_input_tokens
            if cache_input_token!=0:
                logger.debug("LLM prompt cache created %s input tokens.", cache_input_token)
        except AttributeError:
            pass
        try:
            cached_tokens=response.usage.prompt_tokens_details.cached_tokens
            if cached_tokens!=0:
                logger.debug("LLM request reused %s cached input tokens.", cached_tokens)
        except AttributeError:
            pass
        return content.strip() if content else None
    except (AttributeError, IndexError, TypeError) as exc:
        logger.warning("The content of the LLM response could not be extracted (%s).", type(exc).__name__)
        return None
