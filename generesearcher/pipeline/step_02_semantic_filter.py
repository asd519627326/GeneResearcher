# src/pipeline/step_02_semantic_filter.py
import logging
import hashlib
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
import asyncio
from pathlib import Path
import re
from collections import defaultdict

from ..core.data_models import AnalysisData, SentenceEmbeddingRecord
from ..connectors.embedding_connector import EmbeddingManager
from ..connectors.milvus_connector import MilvusManager
from ..core.data_io import AnalysisSaver
from ..core.naming import analysis_directory_name
from ..logging_utils import progress_enabled

logger = logging.getLogger(__name__)

# --- 内部辅助函数 ---

def _load_and_validate_spacy_model(model_name: str) -> Any:
    """Load the required model and explain how to install it if unavailable."""
    from ..runtime_dependencies import load_scispacy_model

    return load_scispacy_model(model_name)

def _segment_text_with_ids(text: str, gene: str, pmid: str, spacy_nlp: Any) -> List[Dict[str, Any]]:
    """将文本分割成带ID的段落和句子。"""
    if not text or not text.strip():
        return []

    structured_sentences = []
    paragraphs = text.split('\n\n')
    sentence_global_index = 0

    for i, para_text in enumerate(paragraphs):
        para_text = para_text.strip()
        if not para_text:
            continue
        
        paragraph_id = f"{gene}_{pmid}_par_{i}"
        doc = spacy_nlp(para_text)
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if sent_text:
                sentence_id = f"{gene}_{pmid}_sen_{sentence_global_index}"
                structured_sentences.append({
                    "gene": gene,
                    "pmid": pmid,
                    "paragraph_id": paragraph_id,
                    "sentence_id": sentence_id,
                    "text": sent_text,
                })
                sentence_global_index += 1
    return structured_sentences


def _upserted_entity_count(result: Any, submitted_count: int) -> int:
    """Read Milvus's upsert count, falling back to the successfully submitted batch size."""
    if isinstance(result, dict):
        count = result.get("upsert_count")
    else:
        count = getattr(result, "upsert_count", None)
    return int(count) if count is not None else submitted_count


async def _process_and_embed_paper(
    paper_row, 
    spacy_nlp, 
    embedding_manager: EmbeddingManager
) -> List[SentenceEmbeddingRecord]:
    """
    Processes a database-backed paper row, segments text, gets embeddings,
    and returns sentence embedding records.
    """
    gene, pmid = paper_row.Gene, str(paper_row.PMID)

    # Internal source text is hydrated from Milvus (or freshly returned by Step 1).
    # The sibling TXT file is only a user-facing export and is not a pipeline input.
    abstract = getattr(paper_row, "abstract", "")
    full_text = getattr(paper_row, "full_text", "")
    abstract = "" if pd.isna(abstract) else str(abstract)
    full_text = "" if pd.isna(full_text) else str(full_text)
    
    combined_text = f"{abstract}\n\n{full_text}".strip()

    if not combined_text:
        logger.warning("Skipping PMID %s / gene %s because no abstract or full text is available.", pmid, gene)
        return []

    # Segment into sentences
    sentences_data = _segment_text_with_ids(combined_text, gene, pmid, spacy_nlp)
    if not sentences_data:
        logger.warning("No sentences were produced from PMID %s / gene %s article text.", pmid, gene)
        return []

    # Get embeddings in a batch
    texts = [s['text'] for s in sentences_data]
    embeddings = await asyncio.to_thread(
        embedding_manager.get_embeddings, texts, text_type='document'
    )

    if embeddings is None or len(embeddings) != len(sentences_data):
        actual_count = 0 if embeddings is None else len(embeddings)
        raise RuntimeError(
            f"Embedding failed for {gene}_{pmid}: generated {actual_count} vectors "
            f"for {len(sentences_data)} sentences. Stopping ingestion to prevent "
            "a successful run with missing sentence vectors."
        )

    # Create sentence embedding records for Milvus.
    milvus_entities = []
    for i, sentence in enumerate(sentences_data):
        entity = SentenceEmbeddingRecord(
            sentence_id=sentence['sentence_id'],
            paragraph_id=sentence['paragraph_id'],
            gene=sentence['gene'],
            pmid=sentence['pmid'],
            text=sentence['text'],
            raw_embedding=embeddings[i]
        )
        milvus_entities.append(entity)
        
    return milvus_entities


# In src/pipeline/step_02_semantic_filter.py

# ... (imports and other functions remain the same) ...

async def run_ingestion_pipeline(analysis_data: AnalysisData, services: Dict[str, Any], show_progress: bool = True):
    """
    Ingestion Step: Processes papers, generates embeddings, and stores them in Milvus
    using a memory-efficient batching strategy.
    """
    embedding_manager = services["embedding_manager"]
    function_name_safe = analysis_directory_name(analysis_data.function_name)
    analysis_dir = Path(analysis_data.project_folder) / f"analysis_{function_name_safe}"
    db_path = analysis_dir / "step2_embeddings.db"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    logger.debug("Using Milvus Lite database '%s'.", db_path)
    milvus_manager = MilvusManager(
        db_path=str(db_path),
        collection_name="biomed_sentences",
        dimension=embedding_manager.dimensions
    )
    
    try:
        spacy_nlp = services.get("spacy_nlp")
        if spacy_nlp is None:
            spacy_nlp = _load_and_validate_spacy_model(analysis_data.scispacy_model)
        papers_df = analysis_data.raw_papers_df
        if papers_df.empty:
            logger.warning("Step 2.1 sentence-level semantic retrieval has no gene-article records to process.")
            return

        article_counts_by_gene = papers_df.groupby("Gene")["PMID"].nunique().to_dict()
        article_breakdown = "; ".join(
            f"{gene}: {int(article_counts_by_gene.get(gene, 0))} article(s)"
            for gene in analysis_data.genes
        )
        logger.info(
            "Step 2.1 sentence-level semantic retrieval is embedding text from %s gene-article record(s): %s.",
            len(papers_df),
            article_breakdown,
        )

        # --- NEW: Define a batch size for writing to the database ---
        BATCH_SIZE = 30 # Number of entities to accumulate before writing
        total_ingested_count = 0
        
        # --- PHASE 1: PRE-COMPUTATION CHECKS (Memory-light) ---
        unique_pmids_in_job = papers_df['PMID'].astype(str).unique().tolist()
        
        # Check which PMIDs already have content in the DB
        pmids_in_db_result = milvus_manager.query(f'pmid in {unique_pmids_in_job}', ['pmid']) or []
        pmids_in_db = {res['pmid'] for res in pmids_in_db_result}
        
        # Check which specific (gene, pmid) associations already exist
        # This is more complex to do in a single bulk query, so we'll check inside the loop,
        # but we will do it before any expensive embedding work.

        # --- PHASE 2: PROCESS AND INGEST IN BATCHES ---
        
        # Lists to hold entities for the current batch
        new_entities_batch = []
        reused_entities_batch = []
        pending_associations = set()
        
        # Cache for content that will be reused within this run
        # This avoids re-fetching the same PMID content multiple times
        content_cache = defaultdict(list) 
        
        SEN_RE = re.compile(r"_sen_(\d+)$")
        PAR_RE = re.compile(r"_par_(\d+)$")

        iterator = papers_df.itertuples()
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(papers_df),
                desc="Sentence-level semantic retrieval",
                unit="article",
            )

        for paper in iterator:
            current_pmid = str(paper.PMID)
            current_gene = paper.Gene
            if show_progress:
                iterator.set_postfix_str(
                    f"{current_gene} | PMID {current_pmid}",
                    refresh=False,
                )
            association_key = (str(current_gene), current_pmid)

            # 1. Check if this exact gene-pmid association already exists
            association_filter = f"pmid == '{current_pmid}' and gene == '{current_gene}'"
            if association_key in pending_associations or milvus_manager.query(association_filter, limit=1):
                logger.debug("Skipping existing gene-PMID association (%s, %s).", current_gene, current_pmid)
                continue

            # 2. Decide whether to embed or reuse
            if current_pmid not in pmids_in_db:
                # This is a completely new paper, needs embedding
                logger.debug("Embedding new article PMID %s for gene '%s'.", current_pmid, current_gene)
                milvus_entities = await _process_and_embed_paper(paper, spacy_nlp, embedding_manager)
                
                if milvus_entities:
                    # Keep canonical rows available until the pending batch is flushed.
                    paper_entities = [entity.to_dict() for entity in milvus_entities]
                    content_cache[current_pmid] = paper_entities
                    new_entities_batch.extend(paper_entities)
                    pmids_in_db.add(current_pmid)
                    pending_associations.add(association_key)

            else:
                # This paper's content exists, we need to reuse it
                logger.debug("Reusing stored article embeddings for PMID %s and gene '%s'.", current_pmid, current_gene)
                
                # Fetch content if not already in our local cache for this run
                if current_pmid not in content_cache:
                    existing_content = milvus_manager.query(
                        filter_expression=f"pmid == '{current_pmid}'",
                        output_fields=["sentence_id", "paragraph_id", "text", "raw_embedding"],
                        limit=16384
                    ) or []
                    # A PMID may already have a copy for multiple genes. Reuse one
                    # record per original sentence/paragraph index to avoid duplicate IDs.
                    canonical = {}
                    for entry in existing_content:
                        sid = entry.get("sentence_id", "")
                        pid = entry.get("paragraph_id", "")
                        sen_match, par_match = SEN_RE.search(sid), PAR_RE.search(pid)
                        if sen_match and par_match:
                            canonical[(sen_match.group(1), par_match.group(1))] = entry
                    content_cache[current_pmid] = list(canonical.values())

                    if not content_cache[current_pmid]:
                        logger.warning(
                            "No reusable sentence embeddings were found for PMID %s while processing gene '%s'.",
                            current_pmid,
                            current_gene,
                        )
                
                # Create new entities by remapping gene
                for entry in content_cache[current_pmid]:
                    sid, pid = entry.get("sentence_id", ""), entry.get("paragraph_id", "")
                    sen_match, par_match = SEN_RE.search(sid), PAR_RE.search(pid)
                    if not sen_match or not par_match: continue
                    
                    sen_idx, par_idx = sen_match.group(1), par_match.group(1)
                    reused_entities_batch.append({
                        "sentence_id": f"{current_gene}_{current_pmid}_sen_{sen_idx}",
                        "paragraph_id": f"{current_gene}_{current_pmid}_par_{par_idx}",
                        "gene": current_gene, "pmid": current_pmid,
                        "text": entry["text"], "raw_embedding": entry["raw_embedding"],
                    })
                if content_cache[current_pmid]:
                    pending_associations.add(association_key)

            # 3. Write to DB if a batch is full
            if len(new_entities_batch) >= BATCH_SIZE:
                batch_count = len(new_entities_batch)
                result = milvus_manager.upsert_batch(new_entities_batch)
                upserted_count = _upserted_entity_count(result, batch_count)
                total_ingested_count += upserted_count
                logger.debug("Milvus upserted %s newly embedded sentence entities.", upserted_count)
                new_entities_batch.clear()

            if len(reused_entities_batch) >= BATCH_SIZE:
                batch_count = len(reused_entities_batch)
                result = milvus_manager.upsert_batch(reused_entities_batch)
                upserted_count = _upserted_entity_count(result, batch_count)
                total_ingested_count += upserted_count
                logger.debug("Milvus upserted %s reused sentence entities.", upserted_count)
                reused_entities_batch.clear()
                
        # --- PHASE 3: FINAL WRITE ---
        # Write any remaining entities that didn't fill a full batch
        if new_entities_batch:
            batch_count = len(new_entities_batch)
            result = milvus_manager.upsert_batch(new_entities_batch)
            upserted_count = _upserted_entity_count(result, batch_count)
            total_ingested_count += upserted_count
            logger.debug("Milvus upserted final batch of %s newly embedded sentence entities.", upserted_count)
        
        if reused_entities_batch:
            batch_count = len(reused_entities_batch)
            result = milvus_manager.upsert_batch(reused_entities_batch)
            upserted_count = _upserted_entity_count(result, batch_count)
            total_ingested_count += upserted_count
            logger.debug("Milvus upserted final batch of %s reused sentence entities.", upserted_count)
            
        logger.info(
            "Step 2.1 completed; upserted %s sentence entity(ies).",
            total_ingested_count,
        )
    finally:
        milvus_manager.close()


# In src/pipeline/step_02_semantic_filter.py

async def run_recall_pipeline(
    analysis_data: AnalysisData,
    services: Dict[str, Any],
    minimum_cosine_similarity: float = 0,
    top_k: int = 16384
) -> pd.DataFrame:
    """
    Recall Step: Generates a query vector and searches Milvus for the most relevant sentences.
    Uses a caching mechanism for query vectors.
    """
    logger.info(
        "Step 2.2: retrieving candidate evidence sentences for each gene-specific research question (minimum cosine similarity %s).",
        minimum_cosine_similarity,
    )
    embedding_manager = services["embedding_manager"]

    # ... (setup code is the same) ...
    function_name_safe = analysis_directory_name(analysis_data.function_name)
    analysis_dir = Path(analysis_data.project_folder) / f"analysis_{function_name_safe}"
    db_path = analysis_dir / "step2_embeddings.db"
    saver = AnalysisSaver(analysis_dir)
    
    if not db_path.exists():
        logger.error("Milvus Lite database '%s' was not found; run sentence ingestion first.", db_path)
        return pd.DataFrame()
        
    milvus_manager = MilvusManager(
        db_path=str(db_path),
        collection_name="biomed_sentences",
        dimension=embedding_manager.dimensions
    )
    
    try:
        all_recalled_sentences = []

        model_name = embedding_manager.model_name
        task_description = "Given a biomedical research context, retrieve relevant sentences that answer the context."
        query_ids_by_gene = {}
        formatted_queries_by_gene = {}
        for gene in analysis_data.genes:
            research_question = analysis_data.research_background_template.format(
                gene=gene, function_name=analysis_data.function_name
            )
            query_context = (
                f"Gene: {gene}\n"
                f"Function/Pathway: {analysis_data.function_name}\n"
                f"Research question/background: {research_question}"
            )
            formatted_query = embedding_manager.format_query(query=query_context, task_description=task_description)
            cache_identity = json.dumps({
                "provider": embedding_manager.provider,
                "model": model_name,
                "task_description": task_description,
                # Keep query embeddings gene-scoped even if two configured
                # templates happen to render identical query text.
                "gene": gene,
                "function_name": analysis_data.function_name,
                "query": formatted_query,
            }, ensure_ascii=False, sort_keys=True)
            query_ids_by_gene[gene] = "q_" + hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()
            formatted_queries_by_gene[gene] = formatted_query

        query_ids_to_check = list(query_ids_by_gene.values())
        cached_vectors = milvus_manager.get_query_vectors(query_ids_to_check)
        logger.debug("Found %s cached query vector(s).", len(cached_vectors))
        
        gene_progress = tqdm(
            analysis_data.genes,
            desc="Sentence-level semantic retrieval",
            unit="gene",
            disable=not progress_enabled(logger),
        )
        for gene in gene_progress:
            gene_progress.set_postfix_str(f"target gene: {gene}")
            recalled_before_gene = len(all_recalled_sentences)
            # This will hold the single vector for the current gene, as a list of floats
            query_vector = None 
            query_id = query_ids_by_gene[gene]

            if query_id in cached_vectors:
                logger.debug("Reusing cached query vector for gene '%s'.", gene)
                # The cached vector is already a list of floats
                query_vector = cached_vectors[query_id]
            else:
                logger.debug("No cached query vector for gene '%s'; embedding the query.", gene)
                formatted_query = formatted_queries_by_gene[gene]
                
                new_embedding_list = await asyncio.to_thread(
                    embedding_manager.get_embeddings, [formatted_query], text_type='query'
                )
                
                if (
                    new_embedding_list is None
                    or new_embedding_list.ndim != 2
                    or new_embedding_list.shape != (1, embedding_manager.dimensions)
                ):
                    raise RuntimeError(
                        f"Failed to generate a complete query embedding for gene '{gene}'."
                    )

                # The result is exactly one vector for this gene's query.
                query_vector = new_embedding_list[0].tolist()
                query_data = {
                    "query_id": query_id,
                    "gene": gene,
                    "function_name": analysis_data.function_name,
                    "model_name": model_name,
                    "query_text": formatted_query,
                    "query_embedding": query_vector,
                }
                milvus_manager.save_query_vector(query_data)
            
            # Now, query_vector is either None or a single list of floats
            if query_vector is None:
                raise RuntimeError(
                    f"Could not generate or retrieve the query embedding for gene '{gene}'."
                )
            if len(query_vector) != embedding_manager.dimensions:
                raise RuntimeError(
                    f"Cached query embedding for gene '{gene}' has dimension {len(query_vector)}; "
                    f"expected {embedding_manager.dimensions}."
                )

            # Milvus search expects a list of vectors. We have one vector.
            search_vectors = [query_vector]
            
            # The query vector represents this gene/function/background
            # research question; the scalar filter independently restricts
            # candidate sentences to this gene's article associations.
            search_filter = f"gene == '{gene}'"
            search_results = milvus_manager.search(
                query_vectors=search_vectors, # Pass the correctly formatted list
                filter_expression=search_filter,
                top_k=top_k
            )
            
            for result in search_results:
                entity = result['entity']
                all_recalled_sentences.append({
                    "Gene": entity.get('gene'),
                    "PMID": entity.get('pmid'),
                    "paragraph_id": entity.get('paragraph_id'),
                    "sentence_id": entity.get('sentence_id'),
                    "text": entity.get('text'),
                    "cosine_similarity": result['distance'],
                })
            logger.debug(
                "Step 2.2: gene '%s' research question recalled %s sentence row(s).",
                gene,
                len(all_recalled_sentences) - recalled_before_gene,
            )

        # ... (The rest of the function remains unchanged) ...
        if not all_recalled_sentences:
            logger.warning(
                "Sentence-level semantic retrieval returned no candidate-evidence sentences for target gene(s): %s.",
                ", ".join(analysis_data.genes),
            )
            milvus_manager.save_recall_results(pd.DataFrame())
            return pd.DataFrame()

        recalled_df_all = pd.DataFrame(all_recalled_sentences)
        
        if not recalled_df_all.empty:
            logger.debug("Sorting recalled sentences by document order.")
            try:
                recalled_df_all['paragraph_num'] = recalled_df_all['paragraph_id'].str.extract(r'_par_(\d+)$').astype(int)
                recalled_df_all['sentence_num'] = recalled_df_all['sentence_id'].str.extract(r'_sen_(\d+)$').astype(int)
                recalled_df_all = recalled_df_all.sort_values(
                    by=['Gene', 'PMID', 'paragraph_num', 'sentence_num']
                ).reset_index(drop=True)
                recalled_df_all = recalled_df_all.drop(columns=['paragraph_num', 'sentence_num'])
                logger.debug("Sorting complete.")
            except Exception as e:
                logger.warning(
                    "Could not sort recalled sentences; preserving the current order (%s).",
                    type(e).__name__,
                )

        saver.save_recalled_sentences(recalled_df_all, similarity_threshold=None)
        
        if minimum_cosine_similarity is not None and minimum_cosine_similarity > 0:
            recalled_df_filtered = recalled_df_all[
                recalled_df_all['cosine_similarity'] >= minimum_cosine_similarity
            ].copy()
            saver.save_recalled_sentences(
                recalled_df_filtered,
                similarity_threshold=minimum_cosine_similarity,
            )
            final_df = recalled_df_filtered
        else:
            final_df = recalled_df_all
            
        milvus_manager.save_recall_results(final_df)
        recalled_counts_by_gene = final_df.groupby("Gene").size().to_dict()
        recalled_before_filtering_by_gene = recalled_df_all.groupby("Gene").size().to_dict()
        recalled_breakdown = "; ".join(
            f"{gene}: {int(recalled_counts_by_gene.get(gene, 0))}"
            for gene in analysis_data.genes
        )
        recalled_before_filtering_breakdown = "; ".join(
            f"{gene}: {int(recalled_before_filtering_by_gene.get(gene, 0))}"
            for gene in analysis_data.genes
        )
        logger.info(
            "Step 2.2 recalled %s candidate-evidence sentence row(s) before similarity filtering (%s); retained %s after filtering (%s).",
            len(recalled_df_all),
            recalled_before_filtering_breakdown,
            len(final_df),
            recalled_breakdown,
        )
        return final_df
    finally:
        milvus_manager.close()
