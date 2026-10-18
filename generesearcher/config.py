# src/config.py

import os
import logging
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv

# 导入构建服务所需的具体类
# 使用相对路径从项目内部的其他模块导入
from .core.parsers import PubMedXMLParser, PMCBioCParser
from .connectors.ncbi_connector import NcbiApiConnector
from .connectors.embedding_connector import EmbeddingManager
from .connectors.llm_api import LLMManager
from .connectors.batch_manager import BatchManager
# 获取模块级别的日志记录器
logger = logging.getLogger(__name__)









def initialize_services(env_file: Optional[str | Path] = None) -> dict[str, Any]:
    """
    初始化并返回项目所需的所有核心服务。
    
    这是一个工厂函数 (Factory Function)，它的作用是封装所有服务对象的创建和配置过程。
    这样，应用的其他部分就不需要知道如何构建这些对象，只需调用此函数即可。
    
    Returns:
        A mapping from service names to initialized connector instances.

    Raises:
        RuntimeError: If required configuration or service initialization fails.
    """
    logger.debug("Core service initialization started.")
    
    try:
        # Use an explicit file when supplied. Otherwise, only read .env from
        # the caller's working directory so behavior does not depend on where
        # this installed package happens to live.
        if env_file is not None:
            dotenv_path = Path(env_file).expanduser()
            if not dotenv_path.is_file():
                raise FileNotFoundError(f"Environment file does not exist: {dotenv_path}")
            load_dotenv(dotenv_path=dotenv_path, override=False)
        else:
            dotenv_path = Path.cwd() / ".env"
            if dotenv_path.is_file():
                load_dotenv(dotenv_path=dotenv_path, override=False)

        # 1. Load environment variables (especially ENTREZ_EMAIL).
        entrez_email = os.getenv("ENTREZ_EMAIL")
        if not entrez_email:
            raise ValueError("ENTREZ_EMAIL environment variable not found. Please set it in your .env file.")

        # 2. 创建解析器实例
        # 这是构建 NcbiApiConnector 的依赖项
        logger.debug("Creating parser instances.")
        pubmed_parser = PubMedXMLParser()
        pmc_parser = PMCBioCParser(entrez_email=entrez_email,max_text_length=120000)
        
        # 3. 创建并配置核心的 NCBI 连接器
        # 这是一个依赖注入的例子：我们将解析器实例 "注入" 到连接器中
        logger.debug("Creating the NCBI connector.")
        ncbi_connector = NcbiApiConnector(
            entrez_email=entrez_email,
            pubmed_parser=pubmed_parser,
            pmc_parser=pmc_parser
        )

       # 4. Inicializar EmbeddingManager
        logger.debug("Creating the embedding manager.")
        embedding_manager = EmbeddingManager(
            provider=os.getenv("EMBEDDING_PROVIDER"),
            model_name=os.getenv("EMBEDDING_MODEL_NAME")
        )

        # 6. Initialize LLMManager
        logger.debug("Creating the LLM manager.")
        llm_manager = LLMManager(
            model_name=os.getenv("LLM_MODEL_NAME"),
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL") # It can be None if not defined
        )
        logger.debug("Creating the Batch manager.")
        batch_manager = BatchManager(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL")
        )



        
        # 6. 将所有创建好的服务打包到一个字典中返回
        # 这种方式非常便于未来扩展：如果增加了新的服务，只需在这里添加一个新的键值对即可
        services = {
            "ncbi_connector": ncbi_connector,
            "embedding_manager": embedding_manager,
            "llm_manager": llm_manager,
            "batch_manager": batch_manager,
        }
        logger.info(
            "Core services ready (embedding=%s/%s; LLM=%s).",
            embedding_manager.provider,
            embedding_manager.model_name,
            llm_manager.model_name,
        )
        return services

    except Exception as e:
        # The public API or CLI records this failure once with the run context.
        raise RuntimeError("Core services initialization failed.") from e


    
