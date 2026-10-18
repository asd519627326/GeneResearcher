"""Typed data structures shared by the GeneResearcher pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..runtime_dependencies import DEFAULT_SCISPACY_MODEL


@dataclass
class Paper:
    """A normalized article record returned by a literature source."""

    paper_id: str
    title: str
    authors: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    published_date: Optional[datetime] = None
    url: Optional[str] = None
    updated_date: Optional[datetime] = None
    keywords: List[str] = field(default_factory=list)
    publication_types: List[str] = field(default_factory=list)
    source: str = ""
    abstract: str = ""
    full_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible article mapping."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "doi": self.doi,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "url": self.url,
            "publication_types": self.publication_types,
            "source": self.source,
            "updated_date": self.updated_date.isoformat() if self.updated_date else None,
            "keywords": self.keywords,
            "abstract": self.abstract,
            "full_text": self.full_text,
        }


@dataclass
class SentenceEmbeddingRecord:
    """A sentence and its vector representation stored in Milvus."""

    sentence_id: str
    paragraph_id: str
    gene: str
    pmid: str
    text: str
    raw_embedding: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Return values suitable for insertion into Milvus."""
        return {
            "sentence_id": self.sentence_id,
            "paragraph_id": self.paragraph_id,
            "gene": self.gene,
            "pmid": self.pmid,
            "text": self.text,
            "raw_embedding": self.raw_embedding,
        }


@dataclass
class AnalysisData:
    """Parameters and in-memory outputs for one function-level analysis."""

    genes: List[str] = field(default_factory=list)
    function_name: str = ""
    research_background_template: str = ""
    keywords: List[str] = field(default_factory=list)
    exclude_pmids: List[str] = field(default_factory=list)
    max_papers_per_gene: int = 10
    project_folder: Optional[str] = None
    function_query: Optional[str] = None
    publication_date_before: Optional[str] = None
    gene_aliases: Dict[str, List[str]] = field(default_factory=dict)
    scispacy_model: str = DEFAULT_SCISPACY_MODEL
    retrieval_status: Optional[str] = None
    relevance_filter_status: Optional[str] = None
    raw_papers_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    recalled_sentences_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    relevance_filtered_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    synthesis_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    reporting_data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_persisted_parameters(self, services: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return the canonical, JSON-serializable project parameter mapping."""
        embedding_model_name = None
        llm_model_name = None
        if services:
            embedding_manager = services.get("embedding_manager")
            if embedding_manager and hasattr(embedding_manager, "model_name"):
                embedding_model_name = embedding_manager.model_name
            llm_manager = services.get("llm_manager")
            if llm_manager and hasattr(llm_manager, "model_name"):
                llm_model_name = llm_manager.model_name

        return {
            "genes": self.genes,
            "function_name": self.function_name,
            "research_background_template": self.research_background_template,
            "keywords": self.keywords,
            "function_query": self.function_query,
            "exclude_pmids": self.exclude_pmids,
            "publication_date_before": self.publication_date_before,
            "gene_aliases": self.gene_aliases,
            "scispacy_model": self.scispacy_model,
            "max_papers_per_gene": self.max_papers_per_gene,
            "timestamp": datetime.now().isoformat(),
            "embedding_model": embedding_model_name,
            "llm_model": llm_model_name,
        }
