# src/pipeline/step_05_reporting.py

import asyncio
import logging
import re
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import pandas as pd
from tqdm import tqdm

from ..connectors.milvus_connector import MilvusManager
from ..connectors.llm_api import LLMManager, get_llm_response_content
from ..core.data_models import AnalysisData
from ..core.data_io import AnalysisSaver
from ..core.naming import analysis_directory_name
from ..logging_utils import progress_enabled

# --- Module-level constants ---
logger = logging.getLogger(__name__)
MAX_CONCURRENT_TASKS = 5
MAX_RETRIES = 3
RETRY_DELAY = 2
NO_QUALIFYING_LITERATURE_MARKER = "no qualifying literature passed step 3 relevance filtering"
RELATIONSHIP_LABELS = {
    "gene_vs_function": "gene–function relationship",
    "gene_vs_context": "gene–research-background relationship",
    "function_vs_context": "function–research-background relationship",
}



def _sanitize_id(text: str) -> str:
    """Helper to create safe IDs for Milvus."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', text).strip('_')




def _extract_and_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts and parses JSON from text (Robustly).
    """
    if not text: return None
    
    clean_text = text.strip()
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean_text, re.DOTALL | re.IGNORECASE)
    
    json_str = ""
    if match:
        json_str = match.group(1)
    else:
        try:
            start_idx = clean_text.index('{')
            end_idx = clean_text.rindex('}') + 1
            json_str = clean_text[start_idx:end_idx]
        except ValueError:
            logger.warning("Could not find a JSON object in the model response.")
            return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            return json.loads(json_str, strict=False)
        except Exception as e:
            logger.warning("Could not decode a JSON object from the model response (%s).", type(e).__name__)
            return None

async def _run_final_mechanism_synthesis(
    llm_manager: LLMManager,
    semaphore: asyncio.Semaphore,
    gene: str,
    function_name: str,
    context: str,
    summaries: Dict[str, str],
    detailed_evidence: Dict[str, str],
    on_started: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """
    Performs the final, high-level synthesis for a single gene.
    """
    async with semaphore:
        if on_started is not None:
            on_started()
        relationship_types = tuple(RELATIONSHIP_LABELS)
        missing_relationships = []
        for analysis_type in relationship_types:
            summary = str(summaries.get(analysis_type, "") or "").strip()
            evidence = str(detailed_evidence.get(analysis_type, "") or "").strip()
            has_summary = bool(summary) and NO_QUALIFYING_LITERATURE_MARKER not in summary.casefold()
            has_evidence = bool(evidence) and "no specific evidence found for this category" not in evidence.casefold()
            if not has_summary and not has_evidence:
                missing_relationships.append(analysis_type)

        if len(missing_relationships) == len(relationship_types):
            explanation = (
                f"Insufficient literature evidence is available to generate an overview for {gene}. "
                "No articles passed Step 3 relevance filtering for the gene–function, "
                "gene–research-background, or function–research-background relationship."
            )
            logger.warning("Step 5 has insufficient qualifying evidence for gene '%s'.", gene)
            return {
                "status": "insufficient_data",
                "gene": gene,
                "final_summary": explanation,
                "cot_trace": [],
            }

        gene_func_summary = summaries.get("gene_vs_function", "") 
        gene_context_summary = summaries.get("gene_vs_context", "")
        func_context_summary = summaries.get("function_vs_context", "") 
        gf_evidence = detailed_evidence.get("gene_vs_function", "No specific evidence found for this category.")
        gc_evidence = detailed_evidence.get("gene_vs_context", "No specific evidence found for this category.")
        fc_evidence = detailed_evidence.get("function_vs_context", "No specific evidence found for this category.")
        
        definitions=f"""
**Analysis Definitions:**
    - **Gene:** '{gene}'
    - **Function/Pathway:** '{function_name}'
    - **Research Background:** "{context}"
            """


            
        # --- MODIFIED: Enriched initial_context ---
        initial_context = f"""
**Part 1: Classification Summary Summaries (Primary Information):**
These are the main conclusions you must synthesize.

**Evidence coverage:**
{', '.join(RELATIONSHIP_LABELS[item] for item in missing_relationships) if missing_relationships else 'All three relationship categories have qualifying Step 4 summaries.'}
If a relationship category is listed as missing, explicitly identify it as a limitation in the final overview. Do not infer evidence for a missing category.

    1.1  **Gene-Function/Pathway Relationship Summary:**
        ```
            {gene_func_summary}
        ```
            
    1.2  **Gene-Background Relationship Summary:**
        ```
            {gene_context_summary}
        ```

    1.3  **Function-Background Relationship Summary (for background):**
        ```
            {func_context_summary}
        ```
            
**Part 2: Detailed Supporting Evidence (Reference Material):**
Use this detailed information, categorized below, to add nuance and specific examples to your final synthesis.

    **2.1 Detailed Evidence for Gene-Function:**
        ```
        {gf_evidence}
        ```
        
    **2.2 Detailed Evidence for Gene-Background:**
        ```
        {gc_evidence}
        ```
        
    **2.3 Detailed Evidence for Function-Background:**
        ```
        {fc_evidence}
        ```
These summaries (Part 1) represent the core conceptual relationships that should be integrated, while the detailed evidence (Part 2) provides the experimental context, including study systems (human patients, animal models, cell systems) and specific methods, enabling reconstruction of the scientific process.
        """

        synthesis_questions = [
    f"{definitions}\n\nUsing the categorical summaries in Part 1, outline the conceptual relationships among **Gene**, **Function/Pathway**, and **Research Background**, keeping the narrative neutral and scientific.",
    
    f"Based on the conceptual relationships identified above, extract from Part 2 the specific experimental evidence—methods used, model systems, and observed results—that are directly relevant to answering: '{context}'. Prioritize patient-derived evidence, followed by higher-order animal models, small-animal models, and then cell-based systems.",
    
    f"Wait,evaluate whether the extracted evidence is sufficient to address: '{context}'. If evidence is incomplete or conflicting, clearly state the limitations.",
    
    f"Provide a revised final summary of '{context}' in Markdown format, written as a concise scientific abstract. The summary should integrate the three relationship domains from Part 1 while reconstructing the scientific process based on Part 2 (experimental approaches → key findings → inferred conclusions). Reflect the relative contribution of different experimental systems naturally within the narrative, without explicitly labeling their hierarchy. Avoid headings and provide only scientific content with correctly formatted PMIDs."
]
        
        # ... (el resto de la función (llamada al LLM, etc.) permanece sin cambios) ...
        messages = [{
            "role": "system",
            "content": "You are a molecular biologist who, based solely on the information provided by the user and without incorporating external knowledge, synthesizes an explanation of the regulatory relationships among the user-defined **Gene**, **Function/Pathway**, and **Research Background**, forming a coherent description."
        }]
        current_context = initial_context + "\n\nBased *only* on this information, please answer the following questions sequentially:"

        try:
            for i, question in enumerate(synthesis_questions):
                if i == 0:
                    # 第一轮：Context + Q1 -> 开启缓存
                    full_prompt = f"{current_context}\n\n**Question {i+1}:**\n{question}"
                    message_content = llm_manager.create_cached_content(full_prompt)
                else:
                    # 后续轮次
                    message_content = f"**Question {i+1}:**\n{question}"
                messages.append({"role": "user", "content": message_content})
                response_data = await asyncio.to_thread(
                    llm_manager.call,
                    messages_list=messages,
                    temperature=1,
                    request_context=(
                        f"Gene={gene} Stage=Step5-Overview Round={i + 1}/"
                        f"{len(synthesis_questions)}"
                    ),
                )
                if response_data is None:
                    raise ValueError(f"LLM call failed at step {i+1}")
                step_response_content = get_llm_response_content(response_data)
                if not step_response_content.strip():
                    raise ValueError(f"LLM returned an empty response at step {i+1}")
                messages.append({"role": "assistant", "content": step_response_content})
            final_summary = messages[-1].get("content", "Failed to extract final summary.")
            if missing_relationships:
                missing_labels = ", ".join(
                    RELATIONSHIP_LABELS[item] for item in missing_relationships
                )
                final_summary = (
                    f"{final_summary.strip()}\n\n"
                    f"Evidence limitation: no articles passed Step 3 relevance filtering for: {missing_labels}."
                )
            return {"status": "success", "gene": gene, "final_summary": final_summary.strip(),"cot_trace": messages}
        except Exception as e:
            logger.warning("Final synthesis failed for gene '%s' (%s).", gene, type(e).__name__)
            return {
                "status": "failed_llm_error",
                "gene": gene,
                "final_summary": "",
                "error_type": type(e).__name__,
                "cot_trace": []
            }

async def _run_ranking_cot(
    llm_manager: LLMManager,
    genes: List[str],
    function_name: str,
    background_query: str,
    gene_summaries: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Performs a multi-step CoT to rank genes based on importance.
    """
    background_query = background_query.replace("{function_name}", function_name)
    summaries_text = "\n\n".join([f"--- Gene: {g} ---\n{s}" for g, s in gene_summaries.items()])
    
    messages = [{
        "role": "system", 
        "content": "You are a senior biomedical researcher who can analyze the research value of multiple genes in a specific research context, rank them in descending order of importance, and provide the rationale for the ranking."
    }]
    reasoning_steps =[
    # Step 1: Prior Knowledge Ranking
    f"""
```
    **Research Context:** "{background_query}"
    **Function/Pathway:** "{function_name}"
    **Candidate Genes:** {', '.join(genes)}
```
    Based on your prior knowledge, describe the *potential scientific role* of each gene within the specified function/pathway and research context. 
    Focus on what each gene is known to do biologically and how it may relate to this study.
    """,
    
    # Step 2: Evidence-Based Correction
    f"""
Here are the specific analysis summaries (Overviews) we generated for each gene based on retrieved literature:
    
    ```
    {summaries_text}
    ```
    
    Please re-evaluate and refine the described roles of each gene strictly based on the provided abstract text.
    For each gene, explain its biological function and its position or influence within the pathway as supported by experimental evidence.
    Cite all relevant evidence using the format "(PMID:12345678)" without omitting citations.
    """,
    
    # Step 3: Reflection (Anti-Hallucination)
    "Wait, are you overinterpreting the functional roles or pathway relevance of any gene beyond what the abstract evidence supports?",
    
    # Step 4: Final Verdict
    """    
    Provide a final consolidated description for each gene.
Each description should be self-contained, scientifically precise, and suitable for later comparison or prioritization.
Do not mention ranking, priority, or relative importance.
Use only evidence-supported statements with correctly formatted PMID citations.
"""
    ]

    full_trace = []
    final_response = ""
    try:
        for i, step_prompt in enumerate(reasoning_steps):
            if i == 1:
                message_content = llm_manager.create_cached_content(step_prompt)
            else:
                message_content = step_prompt
            messages.append({"role": "user", "content": message_content})
            
            resp = await asyncio.to_thread(llm_manager.call, messages_list=messages, temperature=1)
            if not resp:
                raise ValueError(f"LLM returned empty response at step {i+1}")
                
            step_response = get_llm_response_content(resp)
            messages.append({"role": "assistant", "content": step_response})
            
            full_trace.append({
                "step": i + 1,
                "prompt": step_prompt.strip(),
                "response": step_response.strip()
            })
            final_response = step_response
            
        return {
            "final_response": final_response,
            "full_trace": full_trace
        }
    except Exception as e:
        logger.warning("Multi-gene ranking analysis failed (%s).", type(e).__name__)
        return None

async def _extract_ranking_json(
    llm_manager: LLMManager,
    ranking_text: str,
    valid_genes: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Extracts structured JSON for ranking from the text analysis.
    """
    if not ranking_text: return None
    
    format_description = """

The JSON output should contain a single field "genes_ranked", which is an array. Each element of the array represents a gene and its ranking information. Each element contains three fields:

1. "gene_symbol":
   A string representing the gene symbol (e.g., "TP53").

2. "rank":
   An integer indicating the position of the gene in the ordering, starting from 1, with 1 representing the most important..
   The rank value must be provided explicitly but should not be referenced or explained directly in the text.

3. "reason":
   A scientifically grounded explanation describing the gene's biological role and its relevance to the research context.
   The explanation must:
   - Describe a specific biological function or mechanism of the gene
   - Clarify how the gene acts within the relevant pathway or process
   - Include at least one inline citation in the format "(PMID:12345678)"
   - Avoid comparative or ordering language (e.g., "ranked higher", "more important", "less significant")

   The text should be written as a standalone scientific statement that justifies the gene's inclusion and placement without referencing numerical order.
Example:
    ```
    {
      "genes_ranked": [
        {
          "gene_symbol": "string",
          "rank": 1,
          "reason": "string"
        },
        ...
      ]
    }
    ```
    """
    
    messages = [
        {"role": "system", "content": "You are a formatting function. You will only output the content in the transformed format and nothing else."},
        {"role": "user", "content": f"**Content**:\n{ranking_text}\n\n**Valid Genes:** {json.dumps(valid_genes)} (Only include these genes)\n\n**Formatting Requirements**:{format_description}"}
    ]
    
    gene_map = {g.lower(): g for g in valid_genes}
    for attempt in range(MAX_RETRIES):
        try:
            resp = await asyncio.to_thread(
                lambda: llm_manager.call(
                    messages_list=messages, 
                    temperature=1, 
                    response_format={"type": "json_object"}
                )
            )
            content = get_llm_response_content(resp)
            parsed = _extract_and_parse_json(content)
            
            if not parsed:
                raise ValueError("Failed to parse JSON string.")
            
            # --- Validation and Sanitization ---
            # 1. Check root key
            if "genes_ranked" not in parsed:
                # Try to fix common root key errors (e.g., "ranking", "genes")
                for key in parsed.keys():
                    if isinstance(parsed[key], list):
                        parsed["genes_ranked"] = parsed[key]
                        break
                else:
                    raise ValueError("Missing 'genes_ranked' array in JSON.")

            cleaned_list = []
            raw_list = parsed["genes_ranked"]
            
            if not isinstance(raw_list, list):
                raise ValueError("'genes_ranked' is not a list.")

            for item in raw_list:
                # 2. Fix key names in items
                # Map common variants to standard keys
                item_keys = {k.lower(): k for k in item.keys()}
                
                # Get Gene Symbol
                g_key = item_keys.get("gene_symbol") or item_keys.get("gene") or item_keys.get("symbol")
                if not g_key: continue
                raw_gene = str(item[g_key]).strip()
                
                # Normalize Gene Symbol
                # Check against valid_genes map
                normalized_gene = gene_map.get(raw_gene.lower())
                if not normalized_gene:
                    # Fuzzy fix: try removing spaces or brackets
                    # But generally, if it's not in our valid list, we skip or keep raw (let's skip to be safe)
                    # Or keep raw if it looks like a gene
                    normalized_gene = raw_gene # Fallback: trust LLM but keep case
                    # (Better: if valid_genes is provided, filter by it)
                    if valid_genes and normalized_gene not in valid_genes:
                         # Try case-insensitive match again just in case
                         pass 
                
                if valid_genes and normalized_gene not in valid_genes:
                     # Final check: maybe LLM halluncinated a gene not in the list
                     continue

                # Get Rank
                r_key = item_keys.get("rank") or item_keys.get("ranking") or item_keys.get("order")
                raw_rank = item.get(r_key, 999)
                try:
                    # Convert string rank to int (e.g., "1" -> 1)
                    rank_val = int(str(raw_rank).strip().strip('"').strip("'"))
                except ValueError:
                    rank_val = 999
                
                # Get Reason
                rs_key = item_keys.get("reason") or item_keys.get("reasoning") or item_keys.get("rationale")
                reason_val = str(item.get(rs_key, "No reason provided."))
                
                cleaned_list.append({
                    "gene_symbol": normalized_gene,
                    "rank": rank_val,
                    "reason": reason_val
                })

                
            
            if not cleaned_list:
                raise ValueError("No valid gene entries found after validation.")
            return {"genes_ranked": cleaned_list}

        except Exception as e:
            logger.warning(
                "Could not extract ranking JSON on attempt %s/%s (%s).",
                attempt + 1,
                MAX_RETRIES,
                type(e).__name__,
            )
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.warning("All Step 5 ranking JSON extraction attempts failed.")
                return None

async def _generate_ranking_table_markdown(
    llm_manager: LLMManager,
    ranking_data: Dict[str, Any]
) -> str:
    """
    Generates a Markdown table summary from the ranking JSON data.
    """
    if not ranking_data or "genes_ranked" not in ranking_data:
        return ""

    json_str = json.dumps(ranking_data, indent=2)
    prompt = f"""
    Using the JSON data below, generate a Markdown table.
```
{json_str}
```
Table columns:
- Rank
- Gene
- Primary Functional Role
- Pathway Level
- Reason

Instructions:
- Use the "rank" field directly for the Rank column.
- Use "gene_symbol" for the Gene column.
- Derive "Primary Functional Role" as a concise phrase summarizing the gene's core biological function described in the reason.
- Derive "Pathway Level" as a short descriptor (e.g., upstream regulator, core effector, transcriptional regulator, modulatory factor), based strictly on the role described in the reason.
- Populate the Reason column using a shortened version of the provided reason text, preserving the key biological mechanism and inline PMID citation(s).
- Do not introduce new biological interpretations, comparisons, or references.
- Do not add or remove genes.
- Output only the Markdown table and nothing else.
"""
    messages = [
    {"role": "system", "content": "You are a helpful assistant that formats data into Markdown tables."},
    {"role": "user", "content": prompt}
    ]
    try:
        resp = await asyncio.to_thread(llm_manager.call, messages_list=messages, temperature=1)
        return get_llm_response_content(resp).strip()
    except Exception as e:
        logger.warning("Could not generate the Step 5 ranking table (%s).", type(e).__name__)
        return ""

async def _run_story_cot(
    llm_manager: LLMManager,
    genes: List[str],
    background_query: str,
    gene_summaries: Dict[str, str],
    ranking_rationale: str
) -> Optional[Dict[str, Any]]:
    """
    Performs a multi-step CoT to generate a biomedical story/narrative 
    that connects all genes to the background.
    """
    if not genes:
        return None

    # Prepare context inputs
    summaries_text = "\n\n".join([f"--- Gene: {g} ---\n{s}" for g, s in gene_summaries.items()])
    
    messages = [{
        "role": "system", 
        "content": "You are a senior biomedical researcher skilled at proposing scientific hypotheses and constructing coherent biological narratives, capable of integrating information from multiple genes to generate research-relevant content."
    }]
    
    reasoning_steps = [
    # Step 1: Prior Knowledge Story (Blind)
    f"""
    **Research Background:** "{background_query}"
    **Genes involved:** {', '.join(genes)}
    
    Based on your prior knowledge, in the context of this study, how might these genes be functionally related or operate within a shared biological framework? Please provide your reasoning.
    """,
    
    # Step 2: Evidence-Based Correction (Incorporating Overviews & Ranking)
    f"""
    Here is the actual analysis derived from the literature:
    
    **Individual Gene Overviews:**
    ```
    {summaries_text}
    ```
    
    **Comparative Ranking Analysis:**
    ```
    {ranking_rationale}
    ```
    
    Strictly based on the provided information and within the context of this study, reorganize and elucidate the interrelationships among these genes and their potential functional connections.
    The comparative ranking is provided only to help structure the narrative flow and must not be used as evidence of biological regulation or causality.
    Make sure to use evidence from the abstract and cite it in the format "(PMID:12345678)", and do not omit any citations.
    """,
    
    # Step 3: Reflection (Anti-Hallucination)
    "Wait—within the context of this study, have you inferred gene–gene relationships or mechanisms that are not directly supported by the cited abstract evidence?",
    
    # Step 4: Final Synthesis
    "Based on the above information, provide a single coherent paragraph synthesizing how these genes are relevant to the research context. The paragraph should describe mechanistic relationships or functional complementarity where supported by evidence, and include inline citations in the format '(PMID:12345678)' without omitting references."
]

    full_trace = []
    final_response = ""
    
    try:
        for i, step_prompt in enumerate(reasoning_steps):
            if i == 1:
                # 这一步包含了 summaries_text 和 ranking_rationale，值得缓存
                message_content = llm_manager.create_cached_content(step_prompt)
            else:
                message_content = step_prompt
            messages.append({"role": "user", "content": message_content})
            
            resp = await asyncio.to_thread(llm_manager.call, messages_list=messages, temperature=1)
            if not resp:
                raise ValueError(f"LLM returned empty response at Story CoT step {i+1}")
            
            step_response = get_llm_response_content(resp)
            messages.append({"role": "assistant", "content": step_response})
            
            full_trace.append({
                "step": i + 1,
                "prompt": step_prompt.strip(),
                "response": step_response.strip()
            })
            final_response = step_response
            
        return {
            "final_response": final_response,
            "full_trace": full_trace
        }

    except Exception as e:
        logger.warning("Multi-gene story synthesis failed (%s).", type(e).__name__)
        return None



        
async def _run_speculation_generation(
    llm_manager: LLMManager,
    genes: List[str],
    background_query: str,
    function_name: str,
    story_text: str,
    gene_summaries: Dict[str, str]  # [新增参数] 传入基因概览
) -> str:
    """
    Generates scientific speculations in a two-step process:
    1. Brainstorming/Reasoning based on Story + Gene Overviews.
    2. Formatting the output into a structured Markdown report.
    """
    if not story_text:
        return ""

    # 准备基因概览文本
    summaries_context = "\n\n".join([f"**Gene: {g}**\n{s}" for g, s in gene_summaries.items()])

    # --- Round 1: Brainstorming & Reasoning ---
    prompt_round_1 = f"""
**Research Context:**
- **Topic:** "{background_query}"
- **Target Function/Pathway:** "{function_name}"
- **Analyzed Genes:** {', '.join(genes)}

**Data Source 1: Global Analysis Narrative (The "Big Picture")**
{story_text}

**Data Source 2: Individual Gene Mechanisms (The "Details")**
{summaries_context}

Task:
Identify 2–4 high-value scientific hypotheses that are suggested (but not yet demonstrated) by the information above and are most worth testing next.

Rules:
- Prioritize hypotheses with potential relevance to therapy, biomarker discovery, or key regulatory mechanisms.
- Every speculation must logically follow from the existing evidence (no wild guesses). (cite PMIDs where relevant).
- You may reinterpret or extend existing relationships, but do not contradict known evidence.
- Clearly indicate speculative elements.
- Do not introduce new genes or pathways.
- Avoid exaggerated or definitive claims.

Output:
Just list your brainstorming points and the scientific logic behind them. Do not worry about final formatting yet.
"""

    messages = [
        {"role": "system", "content": "You are a senior scientist capable of critical thinking and proposing future research directions."},
        {"role": "user", "content": prompt_round_1}
    ]

    try:
        # Round 1 Call (Brainstorming)
        # 统一使用 temperature=1
        resp1 = await asyncio.to_thread(llm_manager.call, messages_list=messages, temperature=1)
        if not resp1: raise ValueError("LLM failed at Speculation Round 1")
        
        content_round_1 = get_llm_response_content(resp1)
        messages.append({"role": "assistant", "content": content_round_1})

        # --- Round 2: Refinement & Formatting ---
        prompt_round_2 = """
Rewrite them into the following Markdown format only.

For each hypothesis:

    ### **Hypothesis X: <Short Title>**
    **Hypothesis:** Concise scientific statement.  
    **Rationale:** Evidence-based justification from the provided content.  
    **Implication:** Scientific or clinical relevance.
Rules:
    - Output Markdown only.
    - No new content or interpretation.
    - Professional, objective, and scientifically precise.
"""
        messages.append({"role": "user", "content": prompt_round_2})

        # Round 2 Call (Formatting)
        # 格式整理阶段同样使用 temperature=1
        resp2 = await asyncio.to_thread(llm_manager.call, messages_list=messages, temperature=1)
        if not resp2: raise ValueError("LLM failed at Speculation Round 2")

        return get_llm_response_content(resp2).strip()

    except Exception as e:
        logger.warning("Scientific speculation generation failed (%s).", type(e).__name__)
        return ""




# =================================================================
# === Main Pipeline Step Function ===
# =================================================================
async def run_reporting_step(analysis_data: AnalysisData, services: Dict[str, Any]) -> AnalysisData:
    """
    Step 5: High-level Synthesis & DB Persistence.
    Generates all report components via LLM and saves them to Milvus 'reporting_collection'.
    """
    if analysis_data.synthesis_df.empty:
        logger.warning("Synthesis data is empty. Skipping.")
        return analysis_data
        
    llm_manager: LLMManager = services["llm_manager"]
    synthesis_df = analysis_data.synthesis_df
    
    # Generate a safe ID suffix for this function
    sanitized_func = _sanitize_id(analysis_data.function_name)
    
    # This list will collect ALL data to be inserted into Milvus
    reporting_entities = []

    # --- 1. Prepare Data & Evidence Maps ---
    logger.debug("Step 5: preparing gene summaries and evidence maps.")
    
    summaries_per_gene = {}
    detailed_evidence_for_llm = {} 
    detailed_evidence_for_html = {} # Key for Step 6 References
    
    # 1.1 Extract Global Background (from Step 4)
    global_bg_row = synthesis_df[
        (synthesis_df['Gene'] == 'GLOBAL') & 
        (synthesis_df['Analysis_Type'] == 'function_vs_context') &
        (synthesis_df['Summary_Level'] == 'final_synthesis')
    ]
    global_bg_text = global_bg_row.iloc[0]['Summary_Text'] if not global_bg_row.empty else ""

    if global_bg_text:
        reporting_entities.append({
            "record_id": f"global_background_{sanitized_func}",
            "analysis_type": "global_background",
            "gene": "GLOBAL",
            "function_name": analysis_data.function_name,
            "model_name": llm_manager.model_name,
            "cot_trace": "",
            "final_result": global_bg_text
        })
    
    # 1.2 Process Gene Data
    for gene in analysis_data.genes:
        summaries_per_gene[gene] = {}
        detailed_evidence_for_llm[gene] = {}
        detailed_evidence_for_html[gene] = {}
        
        df_for_gene = synthesis_df[synthesis_df['Gene'].str.lower() == gene.lower()]
        if df_for_gene.empty: continue
            
        final_synthesis_rows = df_for_gene[df_for_gene['Summary_Level'] == 'final_synthesis']
        if not final_synthesis_rows.empty:
            summaries_per_gene[gene] = final_synthesis_rows.set_index('Analysis_Type')['Summary_Text'].to_dict()

        llm_evidence_buffer = {"gene_vs_function": [], "gene_vs_context": [], "function_vs_context": []}
        pmid_evidence_html_map = {}
        
        for _, row in final_synthesis_rows.iterrows():
            ana_type = row['Analysis_Type']
            evidence_json = str(row.get('Evidence', ''))
            try:
                evidence_dict = json.loads(evidence_json) if evidence_json.strip().startswith('{') else {}
                for pmid, sentences in evidence_dict.items():
                    if pmid not in pmid_evidence_html_map: pmid_evidence_html_map[pmid] = []
                    for s in sentences:
                        if s not in pmid_evidence_html_map[pmid]: pmid_evidence_html_map[pmid].append(s)
                    
                    # For LLM context
                    clean_sentences = [re.sub(r'\s*\(score:[\d\.]+\)', '', s) for s in sentences]
                    formatted_block = f"--- From PMID: {pmid} ---\n" + "\n".join(f"- {s}" for s in clean_sentences)
                    if ana_type in llm_evidence_buffer:
                        llm_evidence_buffer[ana_type].append(formatted_block)
            except json.JSONDecodeError: pass
                
        detailed_evidence_for_llm[gene] = {k: "\n\n".join(v) for k, v in llm_evidence_buffer.items()}
        detailed_evidence_for_html[gene] = pmid_evidence_html_map

    # SAVE EVIDENCE MAP: Convert the whole HTML-ready map to JSON string and save
    reporting_entities.append({
        "record_id": f"evidence_map_{sanitized_func}",
        "analysis_type": "evidence_map",
        "gene": "ALL",
        "function_name": analysis_data.function_name,
        "model_name": "system",
        "cot_trace": "",
        "final_result": json.dumps(detailed_evidence_for_html)
    })

    # --- 2. Parallel Final Mechanism Synthesis (Overview) ---
    logger.info(
        "Step 5 is generating gene overviews for target gene(s): %s.",
        ", ".join(analysis_data.genes),
    )
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    tasks = []
    active_genes = []
    for gene in analysis_data.genes:
        summaries = summaries_per_gene.get(gene, {})
        evidence_dict = detailed_evidence_for_llm.get(gene, {})
        formatted_context = analysis_data.research_background_template.format(gene=gene, function_name=analysis_data.function_name)
        async def run_gene_overview(
            target_gene=gene,
            target_summaries=summaries,
            target_evidence=evidence_dict,
            target_context=formatted_context,
        ):
            started = False

            def mark_started() -> None:
                nonlocal started
                started = True
                active_genes.append(target_gene)
                overview_progress.set_postfix_str(target_gene)

            try:
                return await _run_final_mechanism_synthesis(
                    llm_manager,
                    semaphore,
                    target_gene,
                    analysis_data.function_name,
                    target_context,
                    target_summaries,
                    target_evidence,
                    on_started=mark_started,
                )
            finally:
                if started:
                    active_genes.remove(target_gene)
                    overview_progress.update(1)
                    overview_progress.set_postfix_str(
                        active_genes[-1] if active_genes else "",
                    )

        tasks.append(run_gene_overview())
            
    final_mechanism_results = {} 
    insufficient_genes = []
    
    if tasks:
        with tqdm(
            total=len(tasks),
            desc="Step 5 gene overview",
            unit="gene",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}",
            disable=not progress_enabled(logger),
        ) as overview_progress:
            results = [await future for future in asyncio.as_completed(tasks)]
        failed_results = [
            result for result in results
            if result.get("status") not in {"success", "insufficient_data", "failed_missing_data"}
        ]
        if failed_results:
            failed_genes = sorted({str(result.get("gene", "unknown")) for result in failed_results})
            failure_types = sorted({str(result.get("error_type", result.get("status", "unknown"))) for result in failed_results})
            raise RuntimeError(
                "Step 5 gene overview generation failed for gene(s): "
                f"{', '.join(failed_genes)} (failure type(s): {', '.join(failure_types)})."
            )

        for res in results:
            gene = res['gene']
            final_mechanism_results[gene] = res['final_summary']
            
            if res['status'] in {'insufficient_data', 'failed_missing_data'}:
                insufficient_genes.append(gene)

            if 'cot_trace' in res:
                reporting_entities.append({
                    "record_id": f"overview_{gene}_{sanitized_func}",
                    "analysis_type": "overview",
                    "gene": gene,
                    "function_name": analysis_data.function_name,
                    "model_name": llm_manager.model_name,
                    "cot_trace": json.dumps(res['cot_trace']),
                    "final_result": res.get('final_summary', '')
                })
    
    # Save Insufficient Genes List
    reporting_entities.append({
        "record_id": f"insufficient_genes_{sanitized_func}",
        "analysis_type": "insufficient_genes",
        "gene": "ALL",
        "function_name": analysis_data.function_name,
        "model_name": "system",
        "cot_trace": "",
        "final_result": json.dumps(insufficient_genes)
    })

    # --- 3. Ranking Analysis ---
    all_target_genes = analysis_data.genes
    formatted_background = analysis_data.research_background_template.format(
        gene=", ".join(all_target_genes),
        function_name=analysis_data.function_name,
    )
    ranking_cot_text = ""
    
    summaries_for_global_analysis = {}
    for gene in all_target_genes:
        if gene in insufficient_genes:
            partials = summaries_per_gene.get(gene, {})
            summaries_for_global_analysis[gene] = f"[Partial Info]: {partials.get('gene_vs_function','')} {partials.get('gene_vs_context','')}"
        else:
            summaries_for_global_analysis[gene] = final_mechanism_results.get(gene, "")
            
    if len(all_target_genes) > 1:
        logger.info("Step 5: generating the multi-gene ranking analysis.")

        ranking_result = await _run_ranking_cot(
            llm_manager, all_target_genes, analysis_data.function_name, 
            formatted_background, summaries_for_global_analysis
        )

        if ranking_result:
            ranking_cot_text = ranking_result.get("final_response", "")
            if "full_trace" in ranking_result:
                # Save Ranking CoT
                reporting_entities.append({
                    "record_id": f"ranking_cot_{sanitized_func}",
                    "analysis_type": "ranking_cot",
                    "gene": "GLOBAL",
                    "function_name": analysis_data.function_name,
                    "model_name": llm_manager.model_name,
                    "cot_trace": json.dumps(ranking_result['full_trace']),
                    "final_result": ranking_cot_text
                })

            # Extract & Save Ranking JSON
            if ranking_cot_text:
                ranking_data = await _extract_ranking_json(llm_manager, ranking_cot_text, all_target_genes)
                if ranking_data:
                    reporting_entities.append({
                        "record_id": f"ranking_data_{sanitized_func}",
                        "analysis_type": "ranking_data",
                        "gene": "GLOBAL",
                        "function_name": analysis_data.function_name,
                        "model_name": llm_manager.model_name,
                        "cot_trace": "",
                        "final_result": json.dumps(ranking_data)
                    })
                    
                    # Generate & Save Ranking Markdown Table
                    ranking_table_md = await _generate_ranking_table_markdown(llm_manager, ranking_data)
                    if ranking_table_md:
                        reporting_entities.append({
                            "record_id": f"ranking_table_{sanitized_func}",
                            "analysis_type": "ranking_table",
                            "gene": "GLOBAL",
                            "function_name": analysis_data.function_name,
                            "model_name": llm_manager.model_name,
                            "cot_trace": "",
                            "final_result": ranking_table_md
                        })

        # --- 4. Story Narrative ---
        logger.info("Step 5: generating the multi-gene study narrative.")
        story_result = await _run_story_cot(
            llm_manager, all_target_genes, formatted_background,
            summaries_for_global_analysis, ranking_cot_text
        )

        story_text = ""
        if story_result:
            story_text = story_result.get("final_response", "")
            if "full_trace" in story_result:
                reporting_entities.append({
                    "record_id": f"story_{sanitized_func}",
                    "analysis_type": "story_narrative",
                    "gene": "GLOBAL",
                    "function_name": analysis_data.function_name,
                    "model_name": llm_manager.model_name,
                    "cot_trace": json.dumps(story_result['full_trace']),
                    "final_result": story_text
                })

        # --- 5. Scientific Speculation ---
        if story_text:
            logger.info("Step 5: generating research hypotheses.")
            speculation_text = await _run_speculation_generation(
                llm_manager, all_target_genes, formatted_background,
                analysis_data.function_name, story_text, summaries_for_global_analysis
            )
            if speculation_text:
                reporting_entities.append({
                    "record_id": f"speculation_{sanitized_func}",
                    "analysis_type": "speculation",
                    "gene": "GLOBAL",
                    "function_name": analysis_data.function_name,
                    "model_name": llm_manager.model_name,
                    "cot_trace": "",
                    "final_result": speculation_text
                })
    else:
        logger.info(
            "Only one target gene is being analyzed; skipping multi-gene ranking, study narrative, and research hypotheses."
        )

    # --- 6. Persist to Milvus ---
    if reporting_entities:
        analysis_dir_name = analysis_directory_name(analysis_data.function_name)
        analysis_dir = Path(analysis_data.project_folder) / f"analysis_{analysis_dir_name}"
        db_path = analysis_dir / "step2_embeddings.db"
        embedding_manager = services.get("embedding_manager")
        dimension = getattr(embedding_manager, "dimensions", None)
        milvus_writer = MilvusManager(
            db_path=str(db_path),
            collection_name="biomed_sentences",
            dimension=dimension,
        )
        try:
            milvus_writer.upsert_reporting_batch(reporting_entities)
            milvus_writer.prune_reporting_results(
                analysis_data.function_name,
                [entity["record_id"] for entity in reporting_entities],
            )
            persisted_ids = {entity["record_id"] for entity in reporting_entities}
            persisted_rows = [
                row for row in milvus_writer.load_reporting_results(analysis_data.function_name)
                if row.get("record_id") in persisted_ids
            ]
        finally:
            milvus_writer.close()

        reporting_df = pd.DataFrame(persisted_rows)
        analysis_data.reporting_data = reporting_df
        AnalysisSaver(analysis_dir).save_reporting_results(reporting_df)

    logger.info(
        "Step 5 saved %s report component(s) to the analysis database.",
        len(analysis_data.reporting_data),
    )
    return analysis_data
