# src/pipeline/step_06_report_generation.py

import logging
import re
import json
import time
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.data_models import AnalysisData
from ..core.naming import analysis_directory_name

# --- Module-level constants ---
logger = logging.getLogger(__name__)

# =================================================================
# === Helper Functions ===
# =================================================================

def _linkify_pmids(text: str) -> str:
    """
    Link canonical PMID identities and the PMCID fallback used when PMID is absent.
    """
    if not isinstance(text, str): return str(text) if text else ""

    def make_link(match: re.Match) -> str:
        label = match.group(0)
        identifier = match.group(2)
        if identifier.upper().startswith("PMC"):
            pmcid = identifier.upper()
            url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
        else:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{identifier}/"
        return f"[{label}]({url})"

    # Do not re-wrap identifiers that are already inside a Markdown link.
    return re.sub(
        r"(?<!\[)(PMID:?\s*(\d{7,8}|PMC\d+))(?!\d)",
        make_link,
        text,
        flags=re.IGNORECASE,
    )

def _downgrade_headers(text: str, target_base_level: int = 3) -> str:
    """
    Adjusts markdown headers in a text block to fit into the report hierarchy.
    """
    if not text: return ""
    
    lines = text.split('\n')
    min_hashes = 99
    has_header = False
    for line in lines:
        match = re.match(r'^(#+)\s', line)
        if match:
            has_header = True
            min_hashes = min(min_hashes, len(match.group(1)))
    
    if not has_header:
        return text

    offset = target_base_level - min_hashes
    
    new_lines = []
    for line in lines:
        match = re.match(r'^(#+)(\s.*)', line)
        if match:
            current_level = len(match.group(1))
            new_level = current_level + offset
            new_level = max(1, min(6, new_level)) 
            new_lines.append(f"{'#' * new_level}{match.group(2)}")
        else:
            new_lines.append(line)
            
    return "\n".join(new_lines)

def _format_speculation_headers(text: str) -> str:
    """
    Specific formatter for Speculation section:
    Detects any markdown header (lines starting with #), converts them to 
    Level 3 (###) and ensures the content is Bold (**).
    """
    if not text: return ""
    
    lines = text.split('\n')
    new_lines = []
    
    for line in lines:
        # Match lines starting with hashes, allow optional space
        match = re.match(r'^(#+)\s*(.*)', line)
        if match:
            content = match.group(2).strip()
            # Clean existing bold markers if they wrap the whole content
            if content.startswith('**') and content.endswith('**'):
                content = content[2:-2].strip()
            
            # Format: ### **Title**
            new_lines.append(f"### **{content}**")
        else:
            new_lines.append(line)
            
    return "\n".join(new_lines)

def _format_ranking_details(json_str: str) -> str:
    """
    Parses the Ranking JSON and formats it into a readable Markdown list.
    """
    try:
        data = json.loads(json_str)
        target_list = []
        if isinstance(data, dict):
            if "genes_ranked" in data:
                target_list = data["genes_ranked"]
            else:
                for v in data.values():
                    if isinstance(v, list):
                        target_list = v
                        break
        elif isinstance(data, list):
            target_list = data

        if not target_list:
            return "> No structured ranking data available."

        md_output = ""
        try:
            target_list.sort(key=lambda ranking_item: int(ranking_item.get("rank", 999)))
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("Could not sort ranking details by rank (%s).", type(exc).__name__)

        for item in target_list:
            rank = item.get("rank", "-")
            gene = item.get("gene_symbol", "Unknown")
            reason = item.get("reason", "").strip()
            md_output += f"**{rank}. {gene}**\n\n{reason}\n\n"
        return md_output
    except Exception as exc:
        logger.warning("Failed to format ranking details (%s).", type(exc).__name__)
        return "> Error formatting ranking details."

def _parse_reporting_data(reporting_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Converts the flat DataFrame from Step 5 into a structured dictionary.
    """
    data_map = {
        'gene_overviews': {},
        'story_narrative': '',
        'speculation': '',
        'ranking_table': '',
        'ranking_data': '',
        'global_background': '',
        'evidence_map': {}
    }
    
    if reporting_df is None or reporting_df.empty:
        return data_map

    for _, row in reporting_df.iterrows():
        a_type = row.get('analysis_type')
        gene = row.get('gene')
        content = row.get('final_result')
        
        if not content: continue

        if a_type == 'overview':
            data_map['gene_overviews'][gene] = content
        elif a_type == 'story_narrative':
            data_map['story_narrative'] = content
        elif a_type == 'speculation':
            data_map['speculation'] = content
        elif a_type == 'ranking_table':
            data_map['ranking_table'] = content
        elif a_type == 'ranking_data':
            data_map['ranking_data'] = content
        elif a_type == 'global_background':
            data_map['global_background'] = content
        elif a_type == 'evidence_map':  # <--- 新增捕获逻辑
            try:
                data_map['evidence_map'] = json.loads(content)
            except (TypeError, ValueError, AttributeError) as exc:
                logger.warning("Could not parse the Step 5 evidence map (%s).", type(exc).__name__)
            
    return data_map


def build_report_markdown(analysis_data: AnalysisData, services: Dict[str, Any]) -> str:
    """
    Builds the final Markdown report text without writing it to disk.
    """
    step5_data = _parse_reporting_data(analysis_data.reporting_data)
    synthesis_df = analysis_data.synthesis_df

    titles_map = {}
    if not analysis_data.raw_papers_df.empty:
        temp_df = analysis_data.raw_papers_df.copy()
        temp_df['PMID'] = temp_df['PMID'].astype(str).str.replace(r'\.0$', '', regex=True)
        titles_map = temp_df.set_index('PMID')['title'].to_dict()

    ranked_genes = []
    ranking_json_str = step5_data.get('ranking_data', '')

    if ranking_json_str:
        try:
            r_data = json.loads(ranking_json_str)
            target_list = r_data.get("genes_ranked", []) if isinstance(r_data, dict) else r_data
            if isinstance(target_list, list):
                target_list.sort(key=lambda ranking_item: int(ranking_item.get("rank", 999)))
                ranked_genes = [
                    ranking_item.get("gene_symbol")
                    for ranking_item in target_list
                    if ranking_item.get("gene_symbol")
                ]
        except Exception as exc:
            logger.warning("Could not parse ranking JSON for report ordering (%s).", type(exc).__name__)

    for g in analysis_data.genes:
        if g not in ranked_genes:
            ranked_genes.append(g)

    target_genes = []
    target_gene_keys = set()
    for gene in analysis_data.genes:
        gene_name = str(gene).strip()
        gene_key = gene_name.casefold()
        if gene_name and gene_key != 'global' and gene_key not in target_gene_keys:
            target_genes.append(gene_name)
            target_gene_keys.add(gene_key)
    is_multi_gene = len(target_gene_keys) > 1

    md_lines = []

    md_lines.append("# <center>Analysis Report</center>\n")

    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    llm_manager = services.get("llm_manager")
    model_name = getattr(llm_manager, "model_name", None)
    if model_name and str(model_name).strip().casefold() in {"system", "unknown"}:
        model_name = None
    if not model_name and not analysis_data.reporting_data.empty:
        model_names = analysis_data.reporting_data.get("model_name", pd.Series(dtype=object))
        model_name = next(
            (
                str(candidate).strip()
                for candidate in model_names
                if pd.notna(candidate)
                and str(candidate).strip().casefold() not in {"system", "unknown"}
                and str(candidate).strip()
            ),
            None,
        )
    model_name = str(model_name).strip() if model_name else "Unknown"

    md_lines.append(f"- Generated Time: {current_time}")
    context_template = analysis_data.research_background_template
    if target_genes:
        if len(target_genes) > 1 and "{gene}" in context_template:
            md_lines.append("- Research Context:")
            for gene in target_genes:
                context = context_template.format(
                    gene=gene,
                    function_name=analysis_data.function_name,
                )
                md_lines.append(f"  - {context}")
        else:
            context = context_template.format(
                gene=target_genes[0],
                function_name=analysis_data.function_name,
            )
            md_lines.append(f"- Research Context: {context}")
    else:
        md_lines.append(f"- Research Context: {context_template}")
    md_lines.append(f"- **Analyzed Genes**: {', '.join(analysis_data.genes)}")
    md_lines.append(f"- **Function/Pathway**: {analysis_data.function_name}")
    md_lines.append(f"- LLM Model: {model_name}\n")
    md_lines.append("---\n")

    if is_multi_gene:
        md_lines.append("# Overall Overview\n")

        story = step5_data.get('story_narrative', '> No narrative available.')
        md_lines.append(story + "\n")

        md_lines.append('<font color="red">*(The following is the reasoning for this analysis based on existing information, for reference only.)*</font>\n')

        speculation = step5_data.get('speculation', '> No speculation available.')
        speculation = _format_speculation_headers(speculation)
        md_lines.append(speculation + "\n")

        md_lines.append("# Gene Ranking Analysis (Importance Contribution)\n")
        md_lines.append("*(Rank the genes based on the potential importance or contribution of their descriptions to this study.)*\n")

        ranking_table = step5_data.get('ranking_table', '')
        if ranking_table:
            md_lines.append(ranking_table + "\n")

        if ranking_json_str:
            md_lines.append(f"{_format_ranking_details(ranking_json_str)}")

    global_bg_text = step5_data.get('global_background', '')
    if global_bg_text:
        bg_lines = global_bg_text.strip().split('\n')
        first_line = bg_lines[0].strip()
        remaining_text = "\n".join(bg_lines[1:]).strip()

        if first_line.startswith('#'):
            clean_title = first_line.lstrip('#').strip()
            md_lines.append(f"# {clean_title}\n")
            md_lines.append(_downgrade_headers(remaining_text, target_base_level=3) + "\n")
        else:
            md_lines.append("# Global Background\n")
            md_lines.append(_downgrade_headers(global_bg_text, target_base_level=3) + "\n")

    md_lines.append("# Subgene Analysis\n")

    overviews = step5_data.get('gene_overviews', {})

    for gene in ranked_genes:
        if gene == 'GLOBAL':
            continue

        md_lines.append(f"## {gene}\n")

        gene_overview = overviews.get(gene, "> No overview generated.")
        md_lines.append("### Overview\n")
        md_lines.append(gene_overview + "\n")

        subset = synthesis_df[
            (synthesis_df['Gene'] == gene) &
            (synthesis_df['Summary_Level'] == 'final_synthesis')
        ]

        type_mapping = [
            ('gene_vs_function', 'Gene Vs Function'),
            ('gene_vs_context', 'Gene Vs Context')
        ]
        if not is_multi_gene:
            type_mapping.append(('function_vs_context', 'Function Vs Research Background'))

        for a_type, default_title in type_mapping:
            row = subset[subset['Analysis_Type'] == a_type]
            if not row.empty:
                content = row.iloc[0]['Summary_Text']
                if not content or "No literature" in content:
                    continue

                c_lines = content.strip().split('\n')
                first_line = c_lines[0].strip()
                rem_text = "\n".join(c_lines[1:]).strip()

                if first_line.startswith('#'):
                    clean_title = first_line.lstrip('#').strip()
                    md_lines.append(f"### {clean_title}\n")
                    md_lines.append(_downgrade_headers(rem_text, target_base_level=4) + "\n")
                else:
                    md_lines.append(f"### {default_title}\n")
                    md_lines.append(_downgrade_headers(content, target_base_level=4) + "\n")

    md_lines.append("# References\n")

    full_evidence_map = step5_data.get('evidence_map', {})
    per_doc_df = synthesis_df[synthesis_df['Summary_Level'] == 'per_document']

    for gene in ranked_genes:
        if gene == 'GLOBAL':
            continue

        gene_docs = per_doc_df[per_doc_df['Gene'] == gene]
        if gene_docs.empty:
            continue

        md_lines.append(f"## {gene}\n")

        gene_ev_map = full_evidence_map.get(gene, {})
        pmid_groups = gene_docs.groupby('PMID')

        for raw_pmid, group in pmid_groups:
            pmid_str = str(raw_pmid).replace('.0', '').strip()
            paper_title = titles_map.get(pmid_str, "Title not found")
            md_lines.append(f"### PMID: {pmid_str} - {paper_title}\n")

            type_order = ["gene_vs_function", "gene_vs_context", "function_vs_context"]
            for a_type in type_order:
                row = group[group['Analysis_Type'] == a_type]
                if not row.empty:
                    summary = row.iloc[0]['Summary_Text']
                    if summary and "No literature" not in summary:
                        pretty_type = a_type.replace('_', ' ').title().replace('Vs', 'Vs')
                        md_lines.append(f"**Analysis: {pretty_type}**\n")
                        md_lines.append(f"{summary}\n")

            sentences = gene_ev_map.get(pmid_str, [])

            if sentences:
                md_lines.append("**Key Evidence Sentences:**\n")

                for sent in sentences:
                    clean_sent = re.sub(r'<[^>]+>', '', sent)
                    clean_sent = re.sub(r'\(score:([\d\.]+)\)', r'**(score:\1)**', clean_sent)
                    md_lines.append(f"- {clean_sent}")

                md_lines.append("\n")

    full_text = "\n".join(md_lines)
    return _linkify_pmids(full_text)

# =================================================================
# === Main Logic ===
# =================================================================

def run_report_generation(analysis_data: AnalysisData, services: Dict[str, Any]) -> AnalysisData:
    project_folder = Path(analysis_data.project_folder)
    directory_suffix = analysis_directory_name(str(analysis_data.function_name))
    analysis_dir = project_folder / f"analysis_{directory_suffix}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    final_text = build_report_markdown(analysis_data, services)
    
    output_path = analysis_dir / "analysis_report.md"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_text)
        logger.info("Markdown report saved to '%s'.", output_path)
    except Exception as exc:
        logger.error("Could not save the Markdown report to '%s' (%s).", output_path, type(exc).__name__)
        raise RuntimeError(f"Failed to save Markdown report: {output_path}") from exc

    return analysis_data
