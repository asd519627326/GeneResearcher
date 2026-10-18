# src/connectors/milvus_connector.py

import logging
import time
import hashlib
import json
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

from ..runtime_dependencies import _suppress_milvus_lite_pkg_resources_warning

with _suppress_milvus_lite_pkg_resources_warning():
    from pymilvus import MilvusClient, exceptions, CollectionSchema, FieldSchema, DataType, MilvusException

logger = logging.getLogger(__name__)


class _MilvusLiteIteratorFallbackFilter(logging.Filter):
    """Filter only PyMilvus's expected client-timestamp fallback notice."""

    _MESSAGE = "failed to get mvccTs from milvus server, use client-side ts instead"

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != self._MESSAGE


@contextmanager
def _suppress_milvus_lite_iterator_fallback_notice():
    iterator_logger = logging.getLogger("pymilvus.orm.iterator")
    notice_filter = _MilvusLiteIteratorFallbackFilter()
    iterator_logger.addFilter(notice_filter)
    try:
        yield
    finally:
        iterator_logger.removeFilter(notice_filter)


class MilvusManager:
    """
    A manager to interact with Milvus Lite, handling both sentence and query collections.
    """
    def __init__(self, db_path: str, collection_name: str, dimension: Optional[int]):
        self.db_path = db_path
        self.dimension = dimension
        self.max_retry = 2
        
        # --- Define names for both collections ---
        self.sentence_collection_name = collection_name
        self.query_collection_name = f"{collection_name}_queries"
        self.llm_analysis_collection_name = f"{collection_name}_llm_analysis"
        self.synthesis_collection_name = f"{collection_name}_synthesis"
        self.reporting_collection_name = f"{collection_name}_reporting_analysis"
        self.paper_data_collection_name = f"{collection_name}_paper_data"
        self.recall_results_collection_name = f"{collection_name}_recall_results"
        self.client = None
        

        try:
            logger.debug("Initializing Milvus Lite client for database '%s'.", self.db_path)
            # Milvus Lite 2.5 imports pkg_resources during local client startup;
            # suppress only its known deprecation notice around that import path.
            with _suppress_milvus_lite_pkg_resources_warning():
                self.client = MilvusClient(
                    self.db_path,
                    # Keep the gRPC options scoped to this manager and let close() release
                    # the channel instead of reusing a shared idle Lite connection.
                    dedicated=True,
                    # PyMilvus 3.0.0 otherwise sends idle keepalive pings every 10s.
                    # Milvus Lite's embedded gRPC server can reject that cadence with
                    # GOAWAY ENHANCE_YOUR_CALM (too_many_pings) during long model calls.
                    grpc_options={"grpc.keepalive_permit_without_calls": False},
                )
            logger.debug("Connected to Milvus Lite database '%s'.", self.db_path)
            # --- Setup both collections ---
            self._setup_sentence_collection()
            self._setup_query_collection()
            self._setup_llm_analysis_collection()
            self._setup_synthesis_collection()
            self._setup_reporting_collection()
            self._setup_paper_data_collection()
            self._setup_recall_results_collection()
            logger.debug("Milvus collections are ready for database '%s'.", self.db_path)
        except Exception as e:
            # Collection setup may fail after the local database has been opened.
            # Close that partially initialized client so a retry cannot inherit
            # its file lock.
            self.close()
            raise RuntimeError(
                f"Could not initialize Milvus Lite database '{self.db_path}'. "
                "If the preceding error says the file is open in another program, stop the "
                "other GeneResearcher process or restart its Python/Colab runtime before "
                "retrying. Do not delete or replace the database while it is open. "
                "If no other process is using it, check the pymilvus and milvus-lite installation."
            ) from e

    def __enter__(self):
        if self.client is None:
             raise RuntimeError("MilvusManager was not initialized correctly.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _query_all(self, collection_name: str, output_fields: List[str], filter_expression: str = "") -> List[Dict[str, Any]]:
        """Read every matching row in bounded pages instead of silently truncating at a fixed limit."""
        # Milvus Lite does not return a usable iterator session timestamp here.
        # PyMilvus deliberately falls back to a client-side timestamp; silence
        # only its expected notice during iterator initialization.
        with _suppress_milvus_lite_iterator_fallback_notice():
            iterator = self.client.query_iterator(
                collection_name=collection_name,
                batch_size=1000,
                limit=-1,
                filter=filter_expression,
                output_fields=output_fields,
            )
        rows = []
        try:
            while True:
                page = iterator.next()
                if not page:
                    break
                rows.extend(page)
        finally:
            iterator.close()
        return rows
    
    # --- Methods for Sentence Collection ---

    def _get_sentence_collection_schema(self) -> CollectionSchema:
        """Defines the schema for the main sentence collection."""
        fields = [
            FieldSchema(name="sentence_id", dtype=DataType.VARCHAR, is_primary=True, max_length=1024),
            FieldSchema(name="paragraph_id", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="gene", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="pmid", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="raw_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]
        return CollectionSchema(fields=fields, description="Collection for biomedical sentences", enable_dynamic_field=False)

    def _setup_sentence_collection(self):
        """Creates the sentence collection if it does not exist."""
        collection_name = self.sentence_collection_name
        try:
            if not self.client.has_collection(collection_name):
                if self.dimension is None:
                    logger.debug("Sentence collection is absent and no embedding dimension was supplied; skipping creation.")
                    return
                logger.debug("Creating Milvus collection '%s'.", collection_name)
                schema = self._get_sentence_collection_schema()
                self.client.create_collection(collection_name=collection_name, schema=schema)
                
                # Create vector index
                index_params_vec = self.client.prepare_index_params()
                index_params_vec.add_index(field_name="raw_embedding", metric_type="COSINE")
                self.client.create_index(collection_name, index_params_vec)
                
                # Create scalar indexes
                for field in ["gene", "pmid"]:
                    index_params = self.client.prepare_index_params()
                    index_params.add_index(field_name=field)
                    self.client.create_index(collection_name, index_params)
                logger.debug("Created indexes for Milvus collection '%s'.", collection_name)

            self.client.load_collection(collection_name)
            logger.debug("Loaded sentence collection '%s'.", collection_name)
        except exceptions.MilvusException as e:
            raise RuntimeError(f"Could not initialize sentence collection '{collection_name}'.") from e

    # --- NEW: Methods for Query Collection ---

    def _get_query_collection_schema(self) -> CollectionSchema:
        """Defines the schema for the query vector collection."""
        fields = [
            FieldSchema(name="query_id", dtype=DataType.VARCHAR, is_primary=True, max_length=1024, description="Unique ID: gene_function_model"),
            FieldSchema(name="gene", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="function_name", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="model_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="query_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="query_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
        ]
        return CollectionSchema(fields=fields, description="Collection for query vectors", enable_dynamic_field=False)

    def _setup_query_collection(self):
        """Creates the query collection if it does not exist."""
        collection_name = self.query_collection_name
        try:
            if not self.client.has_collection(collection_name):
                if self.dimension is None:
                    logger.debug("Query collection is absent and no embedding dimension was supplied; skipping creation.")
                    return
                logger.debug("Creating Milvus collection '%s'.", collection_name)
                schema = self._get_query_collection_schema()
                self.client.create_collection(collection_name=collection_name, schema=schema)
                logger.debug("Created Milvus collection '%s'.", collection_name)

            self.client.load_collection(collection_name)
            logger.debug("Loaded query collection '%s'.", collection_name)
        except exceptions.MilvusException as e:
            raise RuntimeError(f"Could not initialize query collection '{collection_name}'.") from e



    def _get_llm_analysis_collection_schema(self) -> CollectionSchema:
        """Defines the schema for the LLM analysis results collection."""
        fields = [
            # Primary key: pmid_gene_analysistype_modelname
            FieldSchema(name="record_id", dtype=DataType.VARCHAR, is_primary=True, max_length=1024),
            FieldSchema(name="pmid", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="gene", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="analysis_type", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="model_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="result", dtype=DataType.VARCHAR, max_length=64),
            # Usar max_length=65535 para campos de texto largos
            FieldSchema(name="json_reason", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="abstract_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="paragraphs_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="reasoning_trace", dtype=DataType.VARCHAR, max_length=65535),
        ]
        return CollectionSchema(fields=fields, description="Collection for Step 3 LLM analysis results", enable_dynamic_field=False)

    def _setup_llm_analysis_collection(self):
        """Creates the LLM analysis collection if it does not exist."""
        collection_name = self.llm_analysis_collection_name
        try:
            if not self.client.has_collection(collection_name):
                logger.debug("Creating Milvus collection '%s'.", collection_name)
                schema = self._get_llm_analysis_collection_schema()
                self.client.create_collection(collection_name=collection_name, schema=schema)
                
                # Crear índices escalares para búsquedas rápidas
                for field in ["pmid", "gene", "analysis_type", "model_name"]:
                    index_params = self.client.prepare_index_params()
                    index_params.add_index(field_name=field)
                    self.client.create_index(collection_name, index_params)
                logger.debug("Created indexes for Milvus collection '%s'.", collection_name)

            self.client.load_collection(collection_name)
            logger.debug("Loaded LLM analysis collection '%s'.", collection_name)
        except exceptions.MilvusException as e:
            raise RuntimeError(f"Could not initialize LLM analysis collection '{collection_name}'.") from e

    def upsert_llm_analysis_batch(self, entities: List[Dict[str, Any]]):
        """Inserts or updates a batch of entities in the LLM analysis collection."""
        if not entities:
            return None
        try:
            entities = self._store_long_text_fields("llm_analysis", entities)
            result = self.client.upsert(self.llm_analysis_collection_name, entities)
            self.client.flush(collection_name=self.llm_analysis_collection_name)
            logger.debug(
                "Upserted and flushed %s entity(ies) to '%s'.",
                len(entities),
                self.llm_analysis_collection_name,
            )
            return result
        except MilvusException as e:
            raise RuntimeError(
                f"Could not upsert LLM analysis records to '{self.llm_analysis_collection_name}'."
            ) from e



    def _get_synthesis_collection_schema(self) -> CollectionSchema:
        """Defines the schema for the Step 4 synthesis results collection."""
        fields = [
            # Primary key: e.g., pmid_gene_type for per-doc, or gene_type_final for final synthesis
            FieldSchema(name="record_id", dtype=DataType.VARCHAR, is_primary=True, max_length=1024),
            FieldSchema(name="pmid", dtype=DataType.VARCHAR, max_length=128), # Nullable for final synthesis
            FieldSchema(name="gene", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="analysis_type", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="model_name", dtype=DataType.VARCHAR, max_length=256),
            # Distinguishes between per-document summaries and final synthesis
            FieldSchema(name="summary_level", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="summary_text", dtype=DataType.VARCHAR, max_length=65535),
            # For per-doc: JSON list of evidence sentences. For final: JSON list of source PMIDs.
            FieldSchema(name="evidence_text", dtype=DataType.VARCHAR, max_length=65535),
        ]
        return CollectionSchema(fields=fields, description="Collection for Step 4 Synthesis results", enable_dynamic_field=False)

    def _setup_synthesis_collection(self):
        """Creates the synthesis collection if it does not exist."""
        collection_name = self.synthesis_collection_name
        try:
            if not self.client.has_collection(collection_name):
                logger.debug("Creating Milvus collection '%s'.", collection_name)
                schema = self._get_synthesis_collection_schema()
                self.client.create_collection(collection_name=collection_name, schema=schema)
                
                # Create scalar indexes for faster filtering
                for field in ["pmid", "gene", "analysis_type", "summary_level"]:
                    index_params = self.client.prepare_index_params()
                    index_params.add_index(field_name=field)
                    self.client.create_index(collection_name, index_params)
                logger.debug("Created indexes for Milvus collection '%s'.", collection_name)

            self.client.load_collection(collection_name)
            logger.debug("Loaded synthesis collection '%s'.", collection_name)
        except exceptions.MilvusException as e:
            raise RuntimeError(f"Could not initialize synthesis collection '{collection_name}'.") from e

    def upsert_synthesis_batch(self, entities: List[Dict[str, Any]]):
        """Inserts or updates a batch of entities in the synthesis collection."""
        if not entities:
            return None
        try:
            entities = self._store_long_text_fields("synthesis", entities)
            result = self.client.upsert(self.synthesis_collection_name, entities)
            self.client.flush(collection_name=self.synthesis_collection_name)
            logger.debug(
                "Upserted and flushed %s entity(ies) to '%s'.",
                len(entities),
                self.synthesis_collection_name,
            )
            return result
        except MilvusException as e:
            raise RuntimeError(
                f"Could not upsert synthesis records to '{self.synthesis_collection_name}'."
            ) from e



    def _get_reporting_collection_schema(self) -> CollectionSchema:
        """Defines the schema for Step 5 reporting analysis traces."""
        fields = [
            # Primary Key: gene_function (for overview) OR ranking_function (for ranking)
            # Format: "overview_{gene}_{function_name}" OR "ranking_{function_name}"
            FieldSchema(name="record_id", dtype=DataType.VARCHAR, is_primary=True, max_length=1024),
            FieldSchema(name="analysis_type", dtype=DataType.VARCHAR, max_length=64), # "overview" or "ranking"
            FieldSchema(name="gene", dtype=DataType.VARCHAR, max_length=256), # Empty for ranking
            FieldSchema(name="function_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="model_name", dtype=DataType.VARCHAR, max_length=256),
            # Store the full CoT conversation as a JSON string
            FieldSchema(name="cot_trace", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="final_result", dtype=DataType.VARCHAR, max_length=65535),
        ]
        return CollectionSchema(fields=fields, description="Collection for Step 5 Reporting CoT traces", enable_dynamic_field=False)

    def _setup_reporting_collection(self):
        """Creates the reporting collection if it does not exist."""
        collection_name = self.reporting_collection_name
        try:
            if not self.client.has_collection(collection_name):
                logger.debug("Creating Milvus collection '%s'.", collection_name)
                schema = self._get_reporting_collection_schema()
                self.client.create_collection(collection_name=collection_name, schema=schema)
                # Index
                index_params = self.client.prepare_index_params()
                index_params.add_index(field_name="analysis_type")
                self.client.create_index(collection_name, index_params)
            self.client.load_collection(collection_name)
        except exceptions.MilvusException as e:
            raise RuntimeError(f"Could not initialize reporting collection '{collection_name}'.") from e

    def _get_paper_data_collection_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="record_id", dtype=DataType.VARCHAR, is_primary=True, max_length=1024),
            FieldSchema(name="article_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="record_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="gene", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="pmid", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="payload", dtype=DataType.VARCHAR, max_length=60000),
        ]
        return CollectionSchema(fields=fields, description="Retrieved paper metadata and text", enable_dynamic_field=False)

    def _setup_paper_data_collection(self):
        collection_name = self.paper_data_collection_name
        if not self.client.has_collection(collection_name):
            self.client.create_collection(collection_name=collection_name, schema=self._get_paper_data_collection_schema())
        self.client.load_collection(collection_name)

    def _get_recall_results_collection_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="record_id", dtype=DataType.VARCHAR, is_primary=True, max_length=1024),
            FieldSchema(name="gene", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="pmid", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="sentence_id", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="payload", dtype=DataType.VARCHAR, max_length=60000),
        ]
        return CollectionSchema(fields=fields, description="Authoritative Step 2 recall results", enable_dynamic_field=False)

    def _setup_recall_results_collection(self):
        collection_name = self.recall_results_collection_name
        if not self.client.has_collection(collection_name):
            self.client.create_collection(collection_name=collection_name, schema=self._get_recall_results_collection_schema())
        self.client.load_collection(collection_name)

    _LONG_TEXT_FIELDS = {
        "llm_analysis": {
            "json_reason": "llm_reason",
            "abstract_text": "llm_abstract",
            "paragraphs_text": "llm_paragraphs",
            "reasoning_trace": "llm_trace",
        },
        "synthesis": {
            "summary_text": "synthesis_summary",
            "evidence_text": "synthesis_evidence",
        },
        "reporting": {
            "cot_trace": "report_trace",
            "final_result": "report_result",
        },
    }

    @staticmethod
    def _overflow_article_id(collection_key: str, record_id: str, field: str) -> str:
        identity = f"{collection_key}\0{record_id}\0{field}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _store_long_text_fields(
        self,
        collection_key: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep large text exact in chunked payload rows; inline Milvus text is byte-limited."""
        fields = self._LONG_TEXT_FIELDS[collection_key]
        normalized_entities = []
        overflow_entities = []
        for original in entities:
            entity = dict(original)
            record_id = str(entity["record_id"])
            for field, record_type in fields.items():
                value = entity.get(field, "")
                value = "" if value is None else str(value)
                overflow_id = self._overflow_article_id(collection_key, record_id, field)
                self.client.delete(
                    collection_name=self.paper_data_collection_name,
                    filter=f'article_id == "{overflow_id}"',
                )
                if len(value.encode("utf-8")) <= 60000:
                    entity[field] = value
                    continue

                entity[field] = ""
                chunks = [value[index:index + 12000] for index in range(0, len(value), 12000)]
                for chunk_index, chunk in enumerate(chunks):
                    overflow_entities.append({
                        "record_id": f"overflow:{overflow_id}:{chunk_index}",
                        "article_id": overflow_id,
                        "record_type": record_type,
                        "gene": self._value_or_empty(entity.get("gene")),
                        "pmid": self._value_or_empty(entity.get("pmid")),
                        "chunk_index": chunk_index,
                        "payload": chunk,
                    })
            normalized_entities.append(entity)

        for start in range(0, len(overflow_entities), 500):
            self.client.upsert(
                self.paper_data_collection_name,
                overflow_entities[start:start + 500],
            )
        if overflow_entities:
            self.client.flush(collection_name=self.paper_data_collection_name)
        return normalized_entities

    def _restore_long_text_fields(
        self,
        collection_key: str,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        fields = self._LONG_TEXT_FIELDS[collection_key]
        record_types = list(fields.values())
        overflow_rows = self._query_all(
            collection_name=self.paper_data_collection_name,
            filter_expression=f"record_type in {json.dumps(record_types)}",
            output_fields=["article_id", "record_type", "chunk_index", "payload"],
        )
        chunks_by_identity = {}
        for overflow in overflow_rows:
            key = (overflow["article_id"], overflow["record_type"])
            chunks_by_identity.setdefault(key, []).append(overflow)

        for row in rows:
            record_id = str(row.get("record_id", ""))
            for field, record_type in fields.items():
                overflow_id = self._overflow_article_id(collection_key, record_id, field)
                chunks = chunks_by_identity.get((overflow_id, record_type), [])
                if chunks:
                    row[field] = "".join(
                        chunk["payload"]
                        for chunk in sorted(chunks, key=lambda item: item["chunk_index"])
                    )
        return rows

    def save_retrieval_status(self, status: str) -> None:
        if status not in {"running", "completed", "no_results"}:
            raise ValueError(f"Unsupported retrieval status: {status}")
        self.client.upsert(self.paper_data_collection_name, [{
            "record_id": "pipeline:retrieval_status",
            "article_id": "pipeline:state",
            "record_type": "retrieval_status",
            "gene": "",
            "pmid": "",
            "chunk_index": 0,
            "payload": status,
        }])
        self.client.flush(collection_name=self.paper_data_collection_name)

    def load_retrieval_status(self) -> Optional[str]:
        rows = self.client.query(
            collection_name=self.paper_data_collection_name,
            ids=["pipeline:retrieval_status"],
            output_fields=["payload"],
        )
        return rows[0].get("payload") if rows else None

    def save_pipeline_stage_status(self, stage: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Persist an aggregate stage state so recovery cannot mistake stale rows for success."""
        if not stage or not stage.replace("_", "").isalnum():
            raise ValueError(f"Invalid pipeline stage name: {stage!r}")
        record_id = f"pipeline:stage_status:{stage}"
        payload = json.dumps({"status": status, "details": details or {}}, ensure_ascii=False)
        self.client.upsert(self.paper_data_collection_name, [{
            "record_id": record_id,
            "article_id": "pipeline:state",
            "record_type": "pipeline_status",
            "gene": "",
            "pmid": "",
            "chunk_index": 0,
            "payload": payload,
        }])
        self.client.flush(collection_name=self.paper_data_collection_name)

    def load_pipeline_stage_status(self, stage: str) -> Optional[Dict[str, Any]]:
        record_id = f"pipeline:stage_status:{stage}"
        rows = self.client.query(
            collection_name=self.paper_data_collection_name,
            ids=[record_id],
            output_fields=["payload"],
        )
        if not rows:
            return None
        try:
            value = json.loads(rows[0].get("payload", "{}"))
        except (TypeError, json.JSONDecodeError):
            logger.error("Invalid persisted status for pipeline stage %s", stage)
            return {"status": "failed", "details": {"reason": "invalid_persisted_status"}}
        return value if isinstance(value, dict) else None

    @staticmethod
    def _value_or_empty(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and value != value:
            return ""
        return str(value)

    @staticmethod
    def _article_id(gene: str, pmid: str) -> str:
        return hashlib.sha256(f"{gene}\0{pmid}".encode("utf-8")).hexdigest()

    def save_retrieved_papers(self, papers_df) -> None:
        """Persist clean source metadata and exact abstract/full-text strings."""
        if papers_df is None or papers_df.empty:
            return

        entities = []
        replacement_ids = {}
        metadata_fields = [
            "title", "authors", "doi", "published_date", "url", "updated_date",
            "keywords", "publication_types", "source", "db_source",
        ]
        for _, row in papers_df.iterrows():
            pmid = self._value_or_empty(row.get("PMID", row.get("paper_id")))
            gene = self._value_or_empty(row.get("Gene"))
            if not pmid or not gene:
                continue

            article_id = self._article_id(gene, pmid)
            metadata = {
                field: row.get(field)
                for field in metadata_fields
                if field in papers_df.columns
            }
            metadata.update({"PMID": pmid, "Gene": gene})
            metadata_payload = json.dumps(metadata, ensure_ascii=False, default=str)
            metadata_chunks = [
                metadata_payload[index:index + 12000]
                for index in range(0, len(metadata_payload), 12000)
            ] or [""]
            article_entities = [
                {
                    "record_id": f"{article_id}:metadata:{chunk_index}",
                    "article_id": article_id,
                    "record_type": "paper_metadata",
                    "gene": gene,
                    "pmid": pmid,
                    "chunk_index": chunk_index,
                    "payload": chunk,
                }
                for chunk_index, chunk in enumerate(metadata_chunks)
            ]

            for field, record_type in (("abstract", "abstract_chunk"), ("full_text", "full_text_chunk")):
                text = self._value_or_empty(row.get(field))
                # 12,000 Unicode code points stay below Milvus's 60,000-byte VARCHAR limit.
                chunks = [text[index:index + 12000] for index in range(0, len(text), 12000)]
                for chunk_index, chunk in enumerate(chunks):
                    article_entities.append({
                        "record_id": f"{article_id}:{record_type}:{chunk_index}",
                        "article_id": article_id,
                        "record_type": record_type,
                        "gene": gene,
                        "pmid": pmid,
                        "chunk_index": chunk_index,
                        "payload": chunk,
                    })

            replacement_ids[article_id] = {entity["record_id"] for entity in article_entities}
            entities.extend(article_entities)

        if not entities:
            return

        # Remove only obsolete text chunks for articles being replaced; stable IDs are upserted.
        for article_id, current_ids in replacement_ids.items():
            existing = self._query_all(
                collection_name=self.paper_data_collection_name,
                filter_expression=f'article_id == "{article_id}"',
                output_fields=["record_id"],
            )
            obsolete_ids = [item["record_id"] for item in existing if item["record_id"] not in current_ids]
            if obsolete_ids:
                self.client.delete(collection_name=self.paper_data_collection_name, ids=obsolete_ids)

        for start in range(0, len(entities), 500):
            self.client.upsert(self.paper_data_collection_name, entities[start:start + 500])
        self.client.flush(collection_name=self.paper_data_collection_name)

    def load_retrieved_papers(self) -> List[Dict[str, Any]]:
        """Rebuild paper rows from Milvus without reading display CSV/TXT files."""
        metadata_rows = self._query_all(
            collection_name=self.paper_data_collection_name,
            filter_expression='record_type == "paper_metadata"',
            output_fields=["article_id", "chunk_index", "payload"],
        )
        if not metadata_rows:
            return []

        metadata_by_article = {}
        for row in metadata_rows:
            metadata_by_article.setdefault(row["article_id"], []).append(row)
        article_ids = list(metadata_by_article)
        chunks_by_article = {article_id: {"abstract_chunk": [], "full_text_chunk": []} for article_id in article_ids}
        for record_type in ("abstract_chunk", "full_text_chunk"):
            chunks = self._query_all(
                collection_name=self.paper_data_collection_name,
                filter_expression=f'record_type == "{record_type}"',
                output_fields=["article_id", "chunk_index", "payload"],
            )
            for chunk in chunks:
                article_id = chunk["article_id"]
                if article_id in chunks_by_article:
                    chunks_by_article[article_id][record_type].append(chunk)

        records = []
        for article_id in article_ids:
            try:
                metadata_payload = "".join(
                    chunk["payload"]
                    for chunk in sorted(metadata_by_article[article_id], key=lambda item: item["chunk_index"])
                )
                metadata = json.loads(metadata_payload)
            except (KeyError, TypeError, json.JSONDecodeError):
                logger.error("Skipping malformed paper metadata record %s", article_id)
                continue

            article_chunks = chunks_by_article[article_id]
            for record_type, field in (("abstract_chunk", "abstract"), ("full_text_chunk", "full_text")):
                ordered = sorted(article_chunks[record_type], key=lambda item: item["chunk_index"])
                metadata[field] = "".join(item["payload"] for item in ordered)
            records.append(metadata)
        return records

    def save_recall_results(self, recalled_df) -> None:
        """Replace the active Step 2 result set in the database."""
        self.client.delete(
            collection_name=self.recall_results_collection_name,
            filter='record_id != ""',
        )
        if recalled_df is None or recalled_df.empty:
            self.client.flush(collection_name=self.recall_results_collection_name)
            return

        entities = []
        for _, row in recalled_df.iterrows():
            gene = self._value_or_empty(row.get("Gene"))
            pmid = self._value_or_empty(row.get("PMID"))
            sentence_id = self._value_or_empty(row.get("sentence_id"))
            identity = hashlib.sha256(f"{gene}\0{sentence_id}".encode("utf-8")).hexdigest()
            payload = {
                "paragraph_id": self._value_or_empty(row.get("paragraph_id")),
                "cosine_similarity": float(row.get("cosine_similarity", 0.0)),
            }
            entities.append({
                "record_id": f"recall:{identity}",
                "gene": gene,
                "pmid": pmid,
                "sentence_id": sentence_id,
                "payload": json.dumps(payload, ensure_ascii=False),
            })

        for start in range(0, len(entities), 500):
            self.client.upsert(self.recall_results_collection_name, entities[start:start + 500])
        self.client.flush(collection_name=self.recall_results_collection_name)

    def load_recall_results(self) -> List[Dict[str, Any]]:
        recall_rows = self._query_all(
            collection_name=self.recall_results_collection_name,
            output_fields=["gene", "pmid", "sentence_id", "payload"],
        )
        if not recall_rows:
            return []

        sentence_ids = list(dict.fromkeys(row["sentence_id"] for row in recall_rows))
        sentence_text = {}
        for start in range(0, len(sentence_ids), 500):
            results = self.client.query(
                collection_name=self.sentence_collection_name,
                ids=sentence_ids[start:start + 500],
                output_fields=["sentence_id", "text"],
            )
            sentence_text.update({row["sentence_id"]: row.get("text", "") for row in results})

        records = []
        for row in recall_rows:
            try:
                payload = json.loads(row["payload"])
            except (KeyError, TypeError, json.JSONDecodeError):
                logger.error("Skipping malformed recall result for sentence %s", row.get("sentence_id"))
                continue
            records.append({
                "Gene": row["gene"],
                "PMID": row["pmid"],
                "paragraph_id": payload.get("paragraph_id", ""),
                "sentence_id": row["sentence_id"],
                "text": sentence_text.get(row["sentence_id"], ""),
                "cosine_similarity": payload.get("cosine_similarity", 0.0),
            })
        return records

    def load_llm_analysis_results(self) -> List[Dict[str, Any]]:
        if not self.client.has_collection(self.llm_analysis_collection_name):
            return []
        rows = self._query_all(
            collection_name=self.llm_analysis_collection_name,
            output_fields=["record_id", "pmid", "gene", "analysis_type", "model_name", "result", "json_reason", "abstract_text", "paragraphs_text", "reasoning_trace"],
        )
        return self._restore_long_text_fields("llm_analysis", rows)

    def load_synthesis_results(self) -> List[Dict[str, Any]]:
        if not self.client.has_collection(self.synthesis_collection_name):
            return []
        rows = self._query_all(
            collection_name=self.synthesis_collection_name,
            output_fields=["record_id", "pmid", "gene", "analysis_type", "model_name", "summary_level", "summary_text", "evidence_text"],
        )
        rows = self._restore_long_text_fields("synthesis", rows)
        return [{
            "Gene": row.get("gene", ""),
            "Analysis_Type": row.get("analysis_type", ""),
            "Summary_Level": row.get("summary_level", ""),
            "PMID": row.get("pmid", ""),
            "Summary_Text": row.get("summary_text", ""),
            "Evidence": row.get("evidence_text", ""),
            "record_id": row.get("record_id", ""),
            "model_name": row.get("model_name", ""),
        } for row in rows]

    def load_reporting_results(self, function_name: str) -> List[Dict[str, Any]]:
        if not self.client.has_collection(self.reporting_collection_name):
            return []
        rows = self._query_all(
            collection_name=self.reporting_collection_name,
            filter_expression=f'function_name == {json.dumps(function_name, ensure_ascii=False)}',
            output_fields=["record_id", "analysis_type", "gene", "function_name", "model_name", "cot_trace", "final_result"],
        )
        return self._restore_long_text_fields("reporting", rows)

    def prune_reporting_results(self, function_name: str, keep_record_ids: List[str]) -> None:
        """Remove stale report rows for this research function after replacements are saved."""
        current_ids = set(keep_record_ids)
        existing = self._query_all(
            collection_name=self.reporting_collection_name,
            filter_expression=f'function_name == {json.dumps(function_name, ensure_ascii=False)}',
            output_fields=["record_id"],
        )
        obsolete_ids = [row["record_id"] for row in existing if row["record_id"] not in current_ids]
        if obsolete_ids:
            self.client.delete(collection_name=self.reporting_collection_name, ids=obsolete_ids)
            self.client.flush(collection_name=self.reporting_collection_name)

    def upsert_reporting_batch(self, entities: List[Dict[str, Any]]):
        """Inserts or updates a batch of entities in the reporting collection."""
        if not entities: return
        try:
            entities = self._store_long_text_fields("reporting", entities)
            result = self.client.upsert(self.reporting_collection_name, entities)
            self.client.flush(collection_name=self.reporting_collection_name)
            logger.debug("Upserted %s entity(ies) to '%s'.", len(entities), self.reporting_collection_name)
            return result
        except MilvusException as e:
            raise RuntimeError(
                f"Could not upsert reporting records to '{self.reporting_collection_name}'."
            ) from e
    
    
    
    def get_query_vectors(self, query_ids: List[str]) -> Dict[str, List[float]]:
        """Fetches existing query vectors from the database in a batch."""
        if not query_ids:
            return {}
        try:
            results = self.client.query(
                collection_name=self.query_collection_name,
                ids=query_ids,
                output_fields=["query_id", "query_embedding"]
            )
            return {item['query_id']: item['query_embedding'] for item in results}
        except MilvusException:
            logger.warning("Could not read cached query vectors; queries will be embedded again.")
            return {}

    def save_query_vector(self, query_data: Dict[str, Any]):
        """Saves a single query vector to the database."""
        try:
            self.client.upsert(self.query_collection_name, [query_data])
            self.client.flush(collection_name=self.query_collection_name)
            logger.debug("Saved cached query vector '%s'.", query_data.get("query_id"))
        except MilvusException:
            logger.warning("Could not save cached query vector '%s'.", query_data.get("query_id"))
    
    # --- Generic Interaction Methods (now specify collection name) ---

    def query(self, filter_expression, output_fields=None, limit=16384):
        # --- 新增：增加保护性检查，防止在空集合上查询 ---
        # 这是一个针对 Milvus Lite 特定 bug 的规避措施
        try:
            if self.count_entities() == 0:
                logger.debug("Query skipped on an empty collection. Returning empty list.")
                return []
        except Exception as e:
            # 如果 count_entities() 本身也失败了，记录日志但继续尝试查询
            logger.warning(
                "Could not check the sentence count before query (%s); attempting the query anyway.",
                type(e).__name__,
            )
        # --- 修改结束 ---

        # This method is for the sentence collection
        return self._execute_with_retry(
            "QUERY",
            self.sentence_collection_name,
            lambda: self.client.query(
                collection_name=self.sentence_collection_name,
                filter=filter_expression,
                output_fields=output_fields or [],
                limit=limit,
            )
        )

    def search(self, query_vectors: List[List[float]], filter_expression: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Searches the sentence collection. This method now correctly maps arguments
        to what the underlying MilvusClient expects.
        """
        output_fields = ["sentence_id", "paragraph_id", "gene", "pmid", "text"]
        
        # --- THIS IS THE FIX ---
        # We call search_with_retry using the correct argument names that
        # the final self.client.search() method expects: 'data' and 'limit'.
        results = self.search_with_retry(
            data=query_vectors,
            filter=filter_expression,
            limit=top_k,
            output_fields=output_fields
        )
        return results[0] if results else []
    
    def upsert_batch(self, entities):
        """Insert or update a batch of entities in the sentence collection."""
        try:
            result = self.client.upsert(self.sentence_collection_name, entities)
            self.client.flush(collection_name=self.sentence_collection_name)
            logger.debug(
                "Upserted and flushed %s entity(ies) to '%s'.",
                len(entities),
                self.sentence_collection_name,
            )
            return result
        except MilvusException as e:
            raise RuntimeError(
                f"Could not upsert sentence records to '{self.sentence_collection_name}'."
            ) from e

    # --- Refactored Retry Logic ---

    def _execute_with_retry(self, operation: str, collection_name: str, operation_callable):
        """Generic retry wrapper for Milvus operations."""
        attempt = 0
        while attempt <= self.max_retry:
            try:
                return operation_callable()
            except MilvusException as e:
                if "Growing segment loss raw data" in str(e):
                    attempt += 1
                    logger.warning(
                        "Milvus %s encountered a growing-segment issue (attempt %s/%s); "
                        "flushing and reloading '%s'.",
                        operation,
                        attempt,
                        self.max_retry,
                        collection_name,
                    )
                    try:
                        self.client.flush(collection_name=collection_name)
                        self.client.load_collection(collection_name=collection_name)
                        time.sleep(0.5)
                    except Exception as inner_e:
                        logger.warning(
                            "Milvus flush/reload recovery failed (%s).",
                            type(inner_e).__name__,
                        )
                    continue
                else:
                    raise RuntimeError(
                        f"Milvus {operation} failed for collection '{collection_name}'."
                    ) from e
        logger.error(
            "Milvus %s failed for collection '%s' after %s attempts.",
            operation,
            collection_name,
            self.max_retry + 1,
        )
        return None

    def search_with_retry(self, **kwargs):
        """
        Specific retry wrapper for the search operation. It now correctly uses
        'filter' and 'limit' which are also arguments of client.search.
        """
        # The kwargs passed here are now already correct: {'data': ..., 'filter': ..., 'limit': ...}
        return self._execute_with_retry(
            "SEARCH",
            self.sentence_collection_name,
            lambda: self.client.search(collection_name=self.sentence_collection_name, **kwargs)
        )

    def count_entities(self) -> Optional[int]:
        try:
            stats = self.client.get_collection_stats(self.sentence_collection_name)
            return int(stats.get('row_count', 0))
        except exceptions.MilvusException:
            logger.warning("Could not read entity count for collection '%s'.", self.sentence_collection_name)
            return None

    def clear_all_collections(self) -> None:
        """Drop every collection in this function's Milvus Lite database."""
        collection_names = self.client.list_collections()
        if not collection_names:
            logger.debug("No Milvus collections to clear in '%s'.", self.db_path)
            return

        for collection_name in collection_names:
            self.client.drop_collection(collection_name=collection_name)
            logger.debug("Cleared Milvus collection '%s'.", collection_name)

        logger.info(
            "Cleared %s Milvus collections from '%s'.",
            len(collection_names),
            self.db_path,
        )



    def close(self):
        """
        Closes the connection to the Milvus database correctly.
        """
        if self.client is None:
            logger.debug("Milvus Lite client is already closed; skipping close.")
            return

        client = self.client
        try:
            # --- FIX 2: Define the list of all collections to be released ---
            all_collections = [
                self.sentence_collection_name,
                self.query_collection_name,
                self.llm_analysis_collection_name,
                self.synthesis_collection_name,
                self.reporting_collection_name,
                self.paper_data_collection_name,
                self.recall_results_collection_name,
            ]
            
            logger.debug("Releasing Milvus collections for database '%s'.", self.db_path)
            
            # --- FIX 3: Iterate over the defined list ---
            for col_name in all_collections:
                # Check if the collection actually exists before trying to release it
                if client.has_collection(col_name):
                    client.release_collection(col_name)
                    logger.debug("Released Milvus collection '%s'.", col_name)
        
        except Exception:
            # If releasing a collection fails, still release the client below.
            logger.exception("Could not release Milvus collections for database '%s'.", self.db_path)
        finally:
            try:
                # PyMilvus 3.0.0 exposes close(); setting self.client to None alone
                # does not unregister the client from its connection manager.
                client.close()
            except Exception:
                logger.exception("Could not close Milvus Lite client for database '%s'.", self.db_path)
            self.client = None
