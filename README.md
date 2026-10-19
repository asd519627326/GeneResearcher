# GeneResearcher

GeneResearcher is a framework for interpreting an individual target gene's function within a defined research context.

[![PyPI version](https://img.shields.io/pypi/v/generesearcher)](https://pypi.org/project/generesearcher/)
[![License](https://img.shields.io/badge/License-BSD--2--Clause-blue.svg)](LICENSE)

> GeneResearcher derives context-specific interpretations and mechanistic clues from available literature. The final interpretation's focus on the research question remains an area for improvement, and proposed molecular mechanisms require experimental validation.

## <img src="assets/icons/flask-conical.svg" width="30" height="30" alt=""> Overview

High-throughput screens and enrichment analyses can identify candidate genes and associated functions, but researchers still need to understand what an individual gene contributes in the biological setting under study. Gene effects can vary—and sometimes have opposing directions—across disease states, cell types, treatments, and molecular conditions, so general functional annotations may not explain a gene's role in a particular context.

GeneResearcher supports this next step in biological interpretation by helping researchers connect a candidate gene and target function to the relevant research context. It brings together literature evidence to clarify context-specific gene function, highlight mechanisms and differences across study conditions, and support hypotheses for experimental investigation.

## <img src="assets/icons/download.svg" width="30" height="30" alt=""> Installation

- Python 3.10–3.12.
- SciSpaCy 0.6.x and a sentence-processing model; the default is `en_core_sci_lg`.
- An NCBI Entrez email, a remote embedding API key, and an OpenAI-compatible LLM API key.
- SciSpaCy 0.6.x requires NumPy 1.x. A dedicated environment is recommended.

Install the package and default language model:

```bash
python -m pip install generesearcher
python -m generesearcher install-spacy-model
```

Other supported models include `en_core_sci_sm`, `en_core_sci_md`, and `en_core_sci_scibert`:

```bash
python -m generesearcher install-spacy-model --model en_core_sci_sm
```

For DashScope embeddings, install the optional extra with `python -m pip install "generesearcher[dashscope]"`. Configure `EMBEDDING_PROVIDER=dashscope` and the relevant API credentials.

To install from source, run these commands from the directory containing `pyproject.toml`:

```bash
git clone https://github.com/asd519627326/GeneResearcher.git
cd GeneResearcher
python -m pip install -e .
python -m generesearcher install-spacy-model
```

## <img src="assets/icons/rocket.svg" width="30" height="30" alt=""> Quick Start

The [Jupyter user guide](notebooks/generesearcher-jupyter-user-guide.ipynb) includes environment setup, credentials, single- and multi-gene examples, and step-by-step inspection.

Set credentials in the process environment, a local `.env` file, notebook secrets, or your deployment secret store. Do not commit real credentials.

```python
import os
from generesearcher import ResearchConfig, run_analysis, save_analysis

os.environ["ENTREZ_EMAIL"] = "your-email@example.org"
os.environ["EMBEDDING_PROVIDER"] = "dashscope"
os.environ["EMBEDDING_API_KEY"] = "your-embedding-api-key"
os.environ["LLM_MODEL_NAME"] = "your-model-name"
os.environ["LLM_API_KEY"] = "your-llm-api-key"
os.environ["LLM_BASE_URL"] = "https://your-llm-endpoint/v1"

# Single-gene example based on the paper's SLAMF6 case study.
config = ResearchConfig(
    genes=["SLAMF6"],
    function_name="T cell receptor signaling",
    research_background_template=(
        "How does {gene} regulate T cell receptor signaling in tumor-reactive "
        "T cells, and how is this related to T cell activation, exhaustion, "
        "antitumor immunity, and cancer immunotherapy response?"
    ),
    publication_date_before="2026-02-11",
)

result = run_analysis(config)
print("Report:", result.report_path)
print(result.analysis_data.raw_papers_df.head())

# Optional: copy the completed analysis to persistent storage.
save_analysis(result, "./results/slamf6_tcr_signaling")
```

This follows the SLAMF6 case study in the paper, with a literature cutoff of 11 February 2026. The paper reports 12 retrieved articles and 990 candidate sentences; live reruns may differ as literature records and services change.

Without `project_folder`, results use temporary storage. Call `result.cleanup()` when you no longer need an unsaved result. A new run for the same function in a selected project folder clears that function's previous database and generated exports.

## <img src="assets/icons/list-ordered.svg" width="30" height="30" alt=""> Ways to Run

**Single gene.** The Quick Start example analyzes `SLAMF6` and T cell receptor signaling in tumor-reactive T cells, antitumor immunity, and cancer immunotherapy. A single-gene run performs gene-level synthesis without an extra cross-gene synthesis.

**Multiple genes.** Keep the target function and research context the same and provide multiple target genes:

```python
multi_gene_config = ResearchConfig(
    genes=[
        "BCL2", "MCL1", "BCL2L1", "BCL2A1", "TP53",
        "MYC", "E2F1", "KRAS", "PTPN11", "FLT3",
    ],
    function_name="cell apoptosis",
    research_background_template=(
        "How does {gene} contribute to acute myeloid leukemia cell survival "
        "and therapeutic resistance through its effects on apoptosis?"
    ),
)
multi_gene_result = run_analysis(multi_gene_config)
```

This follows the paper's acute myeloid leukemia (AML) case study. GeneResearcher analyzes each gene in the same apoptosis and AML context, then adds the multi-gene analyses described in the workflow.

**Run individual stages.** In Jupyter, create one `AnalysisSession` and execute each stage in order, inspecting its in-memory data between cells:

```python
from generesearcher import AnalysisSession

session = AnalysisSession(config)
await session.search_literature()             # Step 1
session.analysis_data.raw_papers_df.head()
await session.retrieve_candidate_sentences()   # Step 2
session.analysis_data.recalled_sentences_df.head()
await session.assess_candidate_evidence()      # Step 3
session.analysis_data.relevance_filtered_df.head()
await session.synthesize_evidence()            # Step 4
session.analysis_data.synthesis_df.head()
await session.analyze_functions()              # Step 5
session.analysis_data.reporting_data.head()
await session.generate_report()                # Step 6
```

Keep the session object and run stages in order. In a regular Python script, place awaited calls inside an async function and run it with `asyncio.run()`.

**Command line.** Create `study.yaml` from `study.example.yaml`, edit the inputs, then run:

```bash
python -m generesearcher run --config study.yaml --save-to ./results/aml_apoptosis
```

## <img src="assets/icons/workflow.svg" width="30" height="30" alt=""> Workflow and Literature APIs

| Stage | Work performed |
| --- | --- |
| 1. Literature retrieval | Search PubMed and PMC using the target gene, target function, and research context; retrieve abstracts and available full text. |
| 2. Candidate evidence identification | Clean and sentence-segment article text, then retrieve sentences relevant to the research question. These sentences form candidate evidence. |
| 3. Candidate evidence assessment | Independently assess support for target gene–target function (*G–F*), target gene–research context (*G–C*), and target function–research context (*F–C*). |
| 4. Evidence synthesis | Compare evidence within each relationship, including findings across study systems and conditions. |
| 5. Interpretation of individual gene function | Integrate the three relationship summaries into a context-specific interpretation. Multi-gene runs add cross-gene analyses. |
| 6. Report | Organize the analysis into a structured report. |

| API or component | How GeneResearcher uses it |
| --- | --- |
| [NCBI Entrez E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) | `ESearch` searches PubMed and PMC separately. `EFetch` retrieves PubMed metadata and abstracts, and provides the final PMC JATS fallback. |
| [PMC BioC API](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/) | First attempt to retrieve PMC full text as BioC XML. |
| [PMC OAI-PMH API](https://pmc.ncbi.nlm.nih.gov/tools/oai/) | Fallback when BioC is unavailable or contains no usable article text; JATS full text is available only for eligible articles. |
| **SciSpaCy** (`en_core_sci_lg`) | Local biomedical NLP pipeline used to segment cleaned abstracts and full texts into sentences. It is a processing library and model, not a web API. |
| **Embedding API** | The manuscript configuration uses Alibaba Cloud `text-embedding-v3` to create 1,024-dimensional sentence and question embeddings; vectors are stored in local Milvus. |
| **LLM API** | Assesses candidate evidence and synthesizes the relationship-level evidence. The OpenAI-compatible endpoint and model are configurable. |

The parsers extract section headings and paragraph text from BioC or JATS XML. If no usable full text is available, GeneResearcher retains available metadata and abstracts.

## <img src="assets/icons/sliders-horizontal.svg" width="30" height="30" alt=""> Configuration

`ResearchConfig` and the study YAML accept the same analysis fields:

| Field | Required | Description |
| --- | --- | --- |
| `genes` | Yes | One or more gene symbols. |
| `function_name` | Yes | Biological function or pathway analyzed in this run. |
| `research_background_template` | Yes | Natural-language research context used to form gene-specific questions; supports `{gene}` and `{function_name}`. The parameter name is retained in the API; its content corresponds to the paper's research context (*C*). |
| `project_folder` | No | Persistent project root for one research context; it can contain multiple target-function analyses. |
| `keywords` | No | Additional search terms. |
| `function_query` | No | Alternative wording for the function in literature searches. |
| `gene_aliases` | No | Additional search aliases keyed by gene symbol. |
| `publication_date_before` | No | Inclusive cutoff in `YYYY-MM-DD` format. |
| `exclude_pmids` | No | PMID values excluded from retrieval. |
| `max_papers_per_gene` | No | Search limit per gene and source; defaults to 10 for PubMed and 10 for PMC, so merged unique results may exceed 10. |
| `scispacy_model` | No | SciSpaCy model; defaults to `en_core_sci_lg`. |

Gene symbols and aliases are alternatives. The target gene, target function, and context-derived keyword groups are joined with AND. Multiple context keywords are joined with OR in PubMed and AND in PMC. PubMed and PMC results are merged, preferring records with usable full text. PMID is the primary article identifier; PMCID is used for PMC retrieval and as a rare fallback when no PMID exists.

Set credentials in `.env` or the process environment. Existing process variables take precedence over `.env`, which is read from the current working directory by default. The Python API accepts `env_file`; the CLI accepts `--env-file`.

| Environment variable | Purpose |
| --- | --- |
| `ENTREZ_EMAIL` | Email address sent with NCBI Entrez requests. |
| `EMBEDDING_PROVIDER` | Remote embedding provider name. |
| `EMBEDDING_MODEL_NAME` | Optional embedding model override. |
| `EMBEDDING_API_KEY` | Remote embedding API key. |
| `LLM_MODEL_NAME` | Language model name. |
| `LLM_API_KEY` | Language-model API key. |
| `LLM_BASE_URL` | OpenAI-compatible LLM endpoint. |
| `EXCLUDE_PMIDS` | Optional comma-separated PMID exclusions. |

See [.env.example](.env.example) for the variable names and example values.

## <img src="assets/icons/database.svg" width="30" height="30" alt=""> Results and API

Each analysis covers one target function and research context for one or more genes. The core interpretation is gene-specific; multi-gene runs also provide cross-gene analyses. A project folder represents a research context and can hold analyses of multiple target functions. The usual Python workflow keeps results in memory and uses temporary storage unless you choose a project folder or save the result.

- `run_analysis(config)` runs all stages synchronously; `run_analysis_async(config)` is the asynchronous entry point.
- `AnalysisSession(config)` runs the six stages individually and exposes in-memory data between stages.
- `AnalysisResult` provides the report path, analysis directory, and analysis data; call `cleanup()` to remove owned temporary files.
- `save_analysis(result, destination)` copies a completed analysis, including its database and generated files, to persistent storage. The target function directory must not already exist.
- `load_analysis(directory)` restores a saved result for inspection without rerunning retrieval or model analysis.

Each run uses a local Milvus Lite database. Structured CSV files are display exports; the database is the authoritative pipeline state. Without `--save-to` or `project_folder`, the CLI uses temporary storage and removes it when the process exits.

GeneResearcher is distributed under the [BSD-2-Clause License](LICENSE). The README icons are from [Lucide](https://github.com/lucide-icons/lucide) and are distributed under the ISC License; see [assets/icons/LICENSE](assets/icons/LICENSE).
