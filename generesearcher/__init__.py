"""GeneResearcher literature-analysis package."""

import logging
import os

# Milvus Lite starts an embedded gRPC server. Keep gRPC's low-severity C-core
# diagnostics from appearing as application errors during normal runs, while
# preserving ERROR/FATAL output and any explicit user setting.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

logging.getLogger(__name__).addHandler(logging.NullHandler())

from .api import AnalysisResult, AnalysisSession, ResearchConfig, run_analysis, run_analysis_async
from .persistence import load_analysis, save_analysis

__all__ = [
    "AnalysisResult",
    "AnalysisSession",
    "ResearchConfig",
    "load_analysis",
    "run_analysis",
    "run_analysis_async",
    "save_analysis",
]

__version__ = "0.1.10"
