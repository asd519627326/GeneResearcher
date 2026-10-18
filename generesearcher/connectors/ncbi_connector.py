# src/connectors/ncbi_connector.py

import logging
import time
import asyncio
import re
import requests
import xml.etree.ElementTree as ET
import copy
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict

try:
    from Bio import Entrez
    BIO_ENTREZ_AVAILABLE = True
except ImportError:
    BIO_ENTREZ_AVAILABLE = False

from ..core.parsers import PubMedXMLParser, PMCBioCParser
from ..core.data_models import Paper

logger = logging.getLogger(__name__)


@dataclass
class PaperFetchResult:
    """Fetched papers and transient source-ID to canonical-ID mappings."""

    papers: List[Paper] = field(default_factory=list)
    source_id_to_paper_id: Dict[str, str] = field(default_factory=dict)


@dataclass
class _HttpResponse:
    """Small urllib response adapter for the PMC BioC and OAI endpoints."""

    status_code: int
    headers: Any
    content: bytes
    url: Optional[str] = None
    retry_delay: Optional[float] = None
    retry_url: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    @property
    def text(self) -> str:
        charset = None
        get_charset = getattr(self.headers, "get_content_charset", None)
        if callable(get_charset):
            charset = get_charset()
        return self.content.decode(charset or "utf-8", errors="replace")

    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(
                f"HTTP request returned status {self.status_code}.",
                response=self,
            )

    def close(self) -> None:
        # The urllib response is fully read and closed before this adapter is returned.
        return None


@dataclass(frozen=True)
class _OaiJatsResult:
    """OAI response and whether its JATS body contains usable prose."""

    xml: Optional[str]
    has_usable_body: bool


class _MetaRefreshParser(HTMLParser):
    """Extract a browser-style HTML meta-refresh instruction, if present."""

    def __init__(self) -> None:
        super().__init__()
        self.delay: Optional[float] = None
        self.url: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "meta":
            return
        attributes = {key.lower(): value or "" for key, value in attrs}
        if attributes.get("http-equiv", "").lower() != "refresh":
            return
        directive = NcbiApiConnector._parse_refresh_directive(
            attributes.get("content", "")
        )
        if directive[0] is not None:
            self.delay, self.url = directive


class NcbiApiConnector:
    MAX_RETRIES = 5
    RETRY_DELAY = 5
    BIOC_MAX_RETRIES = 3
    BIOC_INITIAL_RETRY_DELAY = 3
    BIOC_MAX_RETRY_DELAY = 30
    BIOC_MIN_REQUEST_INTERVAL = 1
    PMC_OAI_MAX_RETRIES = 3
    PMC_OAI_INITIAL_RETRY_DELAY = 3
    PMC_OAI_MAX_RETRY_DELAY = 30
    PMC_OAI_BASE_URL = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
    PMC_OAI_MIN_REQUEST_INTERVAL = 1 / 3
    MAX_FULL_TEXT_LENGTH = 120000

    def __init__(self, entrez_email: str, pubmed_parser: PubMedXMLParser, pmc_parser: PMCBioCParser):
        if not BIO_ENTREZ_AVAILABLE:
            raise ImportError("BioPython (Bio.Entrez) is required for NcbiApiConnector to function.")
        Entrez.email = entrez_email
        self.entrez_email = entrez_email
        self.pubmed_parser = pubmed_parser
        self.pmc_parser = pmc_parser
        self._bioc_request_lock = asyncio.Lock()
        self._bioc_next_request_at = 0.0
        self._pmc_oai_request_lock = asyncio.Lock()
        self._pmc_oai_next_request_at = 0.0
        logger.debug("Initialized the NCBI Entrez connector.")

    @staticmethod
    def _local_name(element: ET.Element) -> str:
        return element.tag.rsplit("}", 1)[-1]

    @staticmethod
    def _get_url_response(
        url: str,
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
    ) -> _HttpResponse:
        """Fetch a PMC endpoint with urllib and adapt status errors for retry handling."""
        request = Request(url, headers=headers or {})
        try:
            with urlopen(request, timeout=timeout) as response:
                return _HttpResponse(
                    status_code=response.status,
                    headers=response.headers,
                    content=response.read(),
                    url=response.geturl(),
                )
        except HTTPError as error:
            return _HttpResponse(
                status_code=error.code,
                headers=error.headers,
                content=error.read(),
                url=error.url,
            )
        except URLError as error:
            raise requests.ConnectionError(
                f"PMC request failed: {error.reason}"
            ) from error
        except (TimeoutError, OSError) as error:
            raise requests.ConnectionError(f"PMC request failed: {error}") from error

    @classmethod
    def _pmc_xml_format(cls, xml_text: str) -> Optional[str]:
        """Return `bioc` or `jats` for parseable PMC XML, otherwise None."""
        try:
            root = ET.fromstring(xml_text)
        except (ET.ParseError, TypeError):
            return None
        root_name = cls._local_name(root).lower()
        names = {cls._local_name(node).lower() for node in root.iter()}
        if root_name in {"collection", "document"} or "passage" in names:
            return "bioc"
        # PMC EFetch can return a valid article-set containing front-matter
        # metadata but no body (for example, when full text is unavailable).
        # Keep that XML so the PMID can still be recovered and PubMed abstract
        # metadata can be used instead of retrying and discarding the record.
        if root_name == "article" or "article" in names:
            return "jats"
        return None

    @staticmethod
    def _usable_full_text(text: Optional[str]) -> bool:
        if not isinstance(text, str) or not text.strip():
            return False
        return text.strip() not in {
            "No structured body prose found.",
            "Failed to parse XML for full text.",
        }

    @staticmethod
    def _parse_refresh_directive(value: Optional[str]) -> tuple[Optional[float], Optional[str]]:
        """Parse an HTTP Refresh or HTML meta-refresh value."""
        if not value:
            return None, None
        delay_text, separator, target_text = value.partition(";")
        try:
            delay = max(0.0, float(delay_text.strip()))
        except (TypeError, ValueError):
            return None, None
        if not separator:
            return delay, None
        match = re.match(r"\s*url\s*=\s*(.*?)\s*$", target_text, flags=re.IGNORECASE)
        if not match:
            return delay, None
        target = match.group(1).strip().strip("\"'") or None
        return delay, target

    @staticmethod
    def _is_bioc_no_healthy_upstream(response: _HttpResponse) -> bool:
        """Detect the proxy response that means the BioC upstream is unavailable."""
        return "no healthy upstream" in response.text.casefold()

    @classmethod
    def _bioc_retry_info(
        cls,
        response: _HttpResponse,
        request_url: str,
        fallback_delay: float,
    ) -> tuple[float, str]:
        """Use Retry-After or browser refresh timing before retrying BioC."""
        delays = [fallback_delay]
        retry_after = cls._parse_retry_after(response.headers.get("Retry-After"))
        if retry_after is not None:
            delays.append(retry_after)

        refresh_delay, refresh_url = cls._parse_refresh_directive(
            response.headers.get("Refresh")
        )
        if refresh_delay is not None:
            delays.append(refresh_delay)

        meta_refresh = _MetaRefreshParser()
        meta_refresh.feed(response.text)
        if meta_refresh.delay is not None:
            delays.append(meta_refresh.delay)
        refresh_url = meta_refresh.url or refresh_url

        retry_url = request_url
        if refresh_url:
            candidate_url = urljoin(response.url or request_url, refresh_url)
            original_host = urlparse(request_url).hostname
            candidate = urlparse(candidate_url)
            if candidate.scheme == "https" and candidate.hostname == original_host:
                retry_url = candidate_url
        return max(delays), retry_url

    async def _extract_pmc_full_text(
        self,
        pmc_id: str,
        xml_text: str,
        publication_types: Optional[List[str]],
        semaphore: asyncio.Semaphore,
    ) -> Optional[str]:
        """Parse BioC/JATS and try alternate PMC JATS sources if BioC has no body."""
        xml_format = self._pmc_xml_format(xml_text)
        if xml_format == "jats":
            return self.pmc_parser.extract_jats_full_text(xml_text, publication_types)

        if xml_format == "bioc":
            extracted = self.pmc_parser.extract_full_text(xml_text, publication_types)
            if self._usable_full_text(extracted):
                return extracted
            logger.debug(
                "BioC returned valid XML without usable body text for PMCID %s; trying PMC OAI-PMH.",
                pmc_id,
            )
            fallback_text = await self._extract_jats_fallback_text(
                pmc_id, publication_types, semaphore
            )
            if fallback_text:
                return fallback_text
            return extracted

        logger.debug("Unrecognized PMC XML format for PMCID %s; checking PMC OAI-PMH.", pmc_id)
        fallback_text = await self._extract_jats_fallback_text(
            pmc_id, publication_types, semaphore
        )
        if fallback_text:
            return fallback_text
        return "Failed to parse XML for full text."

    async def _extract_jats_fallback_text(
        self,
        pmc_id: str,
        publication_types: Optional[List[str]],
        semaphore: asyncio.Semaphore,
    ) -> Optional[str]:
        """Extract usable prose from the shared OAI-PMH → Entrez fallback path."""
        jats_xml = await self._fetch_jats_fallback_xml(
            pmc_id,
            semaphore,
            publication_types=publication_types,
        )
        if not jats_xml:
            return None
        text = self.pmc_parser.extract_jats_full_text(jats_xml, publication_types)
        return text if self._usable_full_text(text) else None

    async def _fetch_jats_fallback_xml(
        self,
        pmc_id: str,
        semaphore: asyncio.Semaphore,
        oai_result_task: Optional[asyncio.Task[_OaiJatsResult]] = None,
        publication_types: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Use OAI-PMH first, then Entrez, preserving parseable metadata XML."""
        oai_result = (
            await oai_result_task
            if oai_result_task is not None
            else await self._fetch_oai_jats_result(pmc_id)
        )
        oai_has_usable_body = oai_result.has_usable_body
        if oai_result.xml and publication_types:
            oai_text = self.pmc_parser.extract_jats_full_text(
                oai_result.xml,
                publication_types,
            )
            oai_has_usable_body = self._usable_full_text(oai_text)
        if oai_has_usable_body:
            logger.warning(
                "PMC BioC did not provide usable full text for PMCID %s; using PMC OAI-PMH.",
                pmc_id,
            )
            return oai_result.xml
        # In the 429 path, OAI-PMH may finish while BioC is still retrying.
        # Log the source switch here, where Entrez is actually called.
        logger.warning(
            "PMC BioC and OAI-PMH did not provide usable full text for PMCID %s; trying Entrez JATS.",
            pmc_id,
        )

        entrez_xml = await self._fetch_entrez_jats_for_pmc(pmc_id, semaphore)
        if entrez_xml:
            entrez_text = self.pmc_parser.extract_jats_full_text(
                entrez_xml,
                publication_types,
            )
            if self._usable_full_text(entrez_text):
                return entrez_xml
        # Keep whichever source returned parseable JATS so its PMID metadata can
        # still be recovered even when neither source has usable body prose.
        return entrez_xml or oai_result.xml

    async def search_ids(
        self,
        query: str,
        database: str,
        max_results: int,
        semaphore: asyncio.Semaphore,
    ) -> List[str]:
        # ... (无变化) ...
        logger.debug(
            "Searching Entrez database '%s' (max_results=%s).",
            database,
            max_results,
        )
        logger.debug("Prepared Entrez query for database '%s' (%s characters).", database, len(query))
        async with semaphore:
            last_error = None
            for attempt in range(self.MAX_RETRIES):
                try:
                    handle = await asyncio.to_thread(
                        Entrez.esearch,
                        db=database,
                        term=query,
                        retmax=max_results,
                        sort="relevance",
                    )
                    record = await asyncio.to_thread(Entrez.read, handle)
                    handle.close()
                    return record.get("IdList", [])
                except Exception as e:
                    last_error = e
                    if attempt < self.MAX_RETRIES - 1:
                        logger.debug(
                            "Entrez search attempt %s/%s failed for database '%s' (%s); retrying.",
                            attempt + 1,
                            self.MAX_RETRIES,
                            database,
                            type(e).__name__,
                        )
                        await asyncio.sleep(self.RETRY_DELAY)
                    else:
                        logger.warning(
                            "Entrez search failed after %s attempts for database '%s' (%s).",
                            self.MAX_RETRIES,
                            database,
                            type(e).__name__,
                        )
            if last_error is not None:
                raise RuntimeError(
                    f"Entrez search for {database} failed after {self.MAX_RETRIES} attempts"
                ) from last_error
        return []

    async def fetch_papers_by_pubmed_ids(
        self,
        pmids: List[str],
        semaphore: asyncio.Semaphore,
        progress_callback: Optional[Callable[[int], Any]] = None,
    ) -> PaperFetchResult:
        # ... (无变化) ...
        unique_pmids = sorted({self._normalize_paper_id(pmid) for pmid in pmids if str(pmid).strip()})
        if not unique_pmids:
            return PaperFetchResult()
        raw_records = await self._fetch_entrez_records(unique_pmids, 'pubmed', semaphore, progress_callback)
        if not raw_records:
            return PaperFetchResult()
        papers = self.pubmed_parser.parse(raw_records, require_abstract=True)
        logger.debug(
            "Parsed %s PubMed article(s) with abstracts from %s fetched record(s).",
            len(papers),
            len(raw_records),
        )
        if len(papers) < len(raw_records):
            logger.warning(
                "Skipped %s PubMed record(s) because they did not produce usable abstract metadata.",
                len(raw_records) - len(papers),
            )
        source_id_to_paper_id = {
            self._normalize_paper_id(paper.paper_id): self._normalize_paper_id(paper.paper_id)
            for paper in papers
            if paper.paper_id
        }
        return PaperFetchResult(papers, source_id_to_paper_id)

    async def fetch_papers_by_pmc_ids(
        self, 
        pmc_ids: List[str], 
        semaphore: asyncio.Semaphore, 
        progress_callback: Optional[Callable[[int], Any]] = None
    ) -> PaperFetchResult:
        # ... (前半部分无变化) ...
        unique_pmc_ids = sorted({self._normalize_pmc_id(pmc_id) for pmc_id in pmc_ids if str(pmc_id).strip()})
        if not unique_pmc_ids:
            return PaperFetchResult()
        logger.debug("Fetching PMC full text for %s unique article ID(s).", len(unique_pmc_ids))
        tasks = {pmc_id: self._fetch_pmc_full_text_xml(pmc_id, semaphore) for pmc_id in unique_pmc_ids}
        task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        id_to_xml_map = {}
        for pmc_id, result in zip(tasks.keys(), task_results):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                logger.warning(
                    "PMC full-text retrieval failed for PMCID %s (%s).",
                    pmc_id,
                    type(result).__name__,
                )
                id_to_xml_map[pmc_id] = None
            else:
                id_to_xml_map[pmc_id] = result
        pmcid_to_pmid_map, pmcids_needing_conversion = self._pre_parse_pmids(id_to_xml_map)
        if pmcids_needing_conversion:
            logger.debug("Converting %s PMC ID(s) to PMID values.", len(pmcids_needing_conversion))
            converted_map = await self._convert_pmcids_to_pmids(pmcids_needing_conversion, semaphore)
            pmcid_to_pmid_map.update(converted_map)
            unresolved_count = len(pmcids_needing_conversion) - len(converted_map)
            if unresolved_count:
                logger.warning(
                    "Could not resolve a PMID for %s of %s PMC article ID(s); PMCID fallback may be used.",
                    unresolved_count,
                    len(pmcids_needing_conversion),
                )
        
        all_pmids_found = list(set(pmcid_to_pmid_map.values()))
        pmid_to_paper_map = {}
        pmids_with_pub_types = {}
        if all_pmids_found:
            logger.debug("Fetching PubMed metadata for %s unique PMID(s) from PMC results.", len(all_pmids_found))
            raw_pubmed_records = await self._fetch_entrez_records(all_pmids_found, 'pubmed', semaphore)
            if raw_pubmed_records:
                papers_with_metadata = self.pubmed_parser.parse(raw_pubmed_records, require_abstract=False)
                pmid_to_paper_map = {p.paper_id: p for p in papers_with_metadata}
                pmids_with_pub_types = {
                    p.paper_id: list(p.publication_types or []) for p in papers_with_metadata
                }

        final_papers = []
        source_id_to_paper_id = {
            self._normalize_pmc_id(pmc_id): self._normalize_paper_id(pmid)
            for pmc_id, pmid in pmcid_to_pmid_map.items()
        }
        for pmc_id in unique_pmc_ids:
            raw_xml = id_to_xml_map.get(pmc_id)
            _p_val = pmcid_to_pmid_map.get(pmc_id)
            pmid = str(_p_val) if _p_val else None
            paper_from_pubmed = pmid_to_paper_map.get(pmid) if pmid else None
            if paper_from_pubmed and raw_xml:
                # ... (无变化) ...
                paper = paper_from_pubmed
                pub_types = pmids_with_pub_types.get(pmid, [])
                extracted_text = await self._extract_pmc_full_text(
                    pmc_id, raw_xml, pub_types, semaphore
                )
                # Keep PubMed metadata and abstract when PMC full text is too
                # large or cannot be parsed after the JATS fallback.
                if not self._usable_full_text(extracted_text):
                    logger.warning(
                        "No usable PMC body text was extracted for PMID %s (PMCID %s); retaining PubMed metadata and abstract.",
                        pmid,
                        pmc_id,
                    )
                    extracted_text = ""
    
                # 2. 正常组装 Paper 对象
                paper = paper_from_pubmed
                paper.full_text = extracted_text
                paper.source = 'pmc'
                if paper.abstract or paper.full_text:
                    final_papers.append(paper)
                    source_id_to_paper_id[self._normalize_pmc_id(pmc_id)] = self._normalize_paper_id(paper.paper_id)
            elif raw_xml:
                # ... (无变化) ...
                if pmid:
                    logger.warning(
                        "PMCID %s maps to PMID %s, but PubMed metadata was unavailable; creating a minimal article record.",
                        pmc_id,
                        pmid,
                    )
                else:
                    logger.warning(
                        "No PMID is available for PMCID %s; using the PMCID fallback identifier.",
                        pmc_id,
                    )
                pub_types = pmids_with_pub_types.get(pmid, []) if pmid else []
                extracted_text = await self._extract_pmc_full_text(
                    pmc_id, raw_xml, pub_types, semaphore
                )
                if not self._usable_full_text(extracted_text):
                    extracted_text = ""
                    logger.warning("No usable body text was extracted for PMC article ID %s.", pmc_id)
                fallback_paper = Paper(
                    paper_id=(pmid or f'PMC{pmc_id}'),
                    title="Title not available from PubMed",
                    url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/" if not pmid else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    source='pmc_no_metadata',
                    full_text=extracted_text
                )
                if fallback_paper.full_text:
                    final_papers.append(fallback_paper)
                    source_id_to_paper_id[self._normalize_pmc_id(pmc_id)] = self._normalize_paper_id(fallback_paper.paper_id)
            elif paper_from_pubmed and (paper_from_pubmed.abstract or paper_from_pubmed.full_text):
                # Keep a usable PubMed record when the PMC request itself did
                # not return XML. Step 1 verifies its canonical PMID against
                # the combined search results before accepting the retrieval.
                metadata_paper = copy.copy(paper_from_pubmed)
                metadata_paper.source = "pmc_metadata"
                final_papers.append(metadata_paper)
                source_id_to_paper_id[self._normalize_pmc_id(pmc_id)] = self._normalize_paper_id(pmid)
                logger.warning(
                    "PMC full text was unavailable for PMCID %s; retaining metadata for PMID %s.",
                    pmc_id,
                    pmid,
                )
            
            if progress_callback:
                progress_callback(1)
        
        logger.debug(
            "Prepared %s usable article record(s) from %s PMC ID(s).",
            len(final_papers),
            len(unique_pmc_ids),
        )
        if len(final_papers) < len(unique_pmc_ids):
            logger.warning(
                "PMC retrieval produced no usable record for %s of %s requested ID(s).",
                len(unique_pmc_ids) - len(final_papers),
                len(unique_pmc_ids),
            )
        return PaperFetchResult(final_papers, source_id_to_paper_id)

    @staticmethod
    def _normalize_pmc_id(value: Any) -> str:
        """Normalize an Entrez PMC search ID to its numeric source ID."""
        normalized = str(value).strip().upper()
        return normalized[3:] if normalized.startswith("PMC") else normalized

    @staticmethod
    def _normalize_paper_id(value: Any) -> str:
        """Normalize the canonical article ID used by later pipeline stages."""
        normalized = str(value).strip()
        if normalized.upper().startswith("PMID:"):
            normalized = normalized[5:].strip()
        if normalized.upper().startswith("PMC"):
            normalized = "PMC" + normalized[3:]
        return normalized

    def _pre_parse_pmids(self, id_to_xml_map: Dict[str, str]) -> (Dict[str, str], List[str]):
        # ... (无变化) ...
        pmcid_to_pmid_map, pmcids_needing_conversion = {}, []
        for pmc_id, raw_xml in id_to_xml_map.items():
            if not raw_xml:
                pmcids_needing_conversion.append(pmc_id)
                continue
            try:
                root = ET.fromstring(raw_xml)
                pmid_value = None
                for node in root.iter():
                    node_name = self._local_name(node).lower()
                    if node_name == "infon" and node.attrib.get("key") == "article-id_pmid":
                        pmid_value = "".join(node.itertext()).strip()
                        break
                    if node_name == "article-id" and node.attrib.get("pub-id-type") == "pmid":
                        pmid_value = "".join(node.itertext()).strip()
                        break
                if pmid_value:
                    pmcid_to_pmid_map[pmc_id] = pmid_value
                else:
                    pmcids_needing_conversion.append(pmc_id)
            except ET.ParseError: pmcids_needing_conversion.append(pmc_id)
        return pmcid_to_pmid_map, pmcids_needing_conversion

    async def _convert_pmcids_to_pmids(self, pmc_ids: List[str], semaphore: asyncio.Semaphore) -> Dict[str, str]:
        # ... (无变化) ...
        if not pmc_ids: return {}
        CHUNK_SIZE = 60
        unique_pmc_ids = list(set(pmc_ids))
        id_chunks = [unique_pmc_ids[i:i + CHUNK_SIZE] for i in range(0, len(unique_pmc_ids), CHUNK_SIZE)]
        async def _fetch_chunk(chunk: List[str]) -> Dict[str, str]:
            ids_with_prefix = [f"PMC{pid}" if not pid.startswith("PMC") else pid for pid in chunk]
            ids_str = ",".join(ids_with_prefix)
            base_url = "https://www.ncbi.nlm.nih.gov/pmc/tools/idconv/api/v1/articles/"
            params = {"tool": "Generesearcher", "email": self.entrez_email, "ids": ids_str, "format": "json"}
            async with semaphore:
                for attempt in range(self.MAX_RETRIES):
                    try:
                        response = await asyncio.to_thread(requests.get, base_url, params=params, timeout=30)
                        response.raise_for_status()
                        data = response.json()
                        chunk_conversion_map = {}
                        if data.get("status") == "ok" and data.get("records"):
                            for record in data["records"]:
                                pmcid_num = record.get("pmcid", "").replace("PMC", "")
                                if pmcid_num and "pmid" in record:
                                    chunk_conversion_map[pmcid_num] = record["pmid"]
                                else:
                                    logger.warning(
                                        "NCBI ID conversion returned no PMID for PMC record %s.",
                                        record.get("pmcid", "unknown"),
                                    )
                        return chunk_conversion_map
                    except requests.RequestException as e:
                        logger.debug(
                            "PMC ID conversion attempt %s/%s failed (%s).",
                            attempt + 1,
                            self.MAX_RETRIES,
                            type(e).__name__,
                        )
                        if attempt < self.MAX_RETRIES - 1:
                            await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
                        else:
                            logger.warning("PMC ID conversion failed after %s attempts; PMCID fallback may be used.", self.MAX_RETRIES)
            return {}
        results_list = await asyncio.gather(*[_fetch_chunk(chunk) for chunk in id_chunks])
        final_conversion_map = {}
        for result_dict in results_list: final_conversion_map.update(result_dict)
        return final_conversion_map
        
    async def _fetch_entrez_records(
        self,
        record_ids: List[str],
        database: str,
        semaphore: asyncio.Semaphore,
        progress_callback: Optional[Callable[[int], Any]] = None,
    ) -> Optional[List[dict]]:
        # ... (无变化) ...
        async with semaphore:
            for attempt in range(self.MAX_RETRIES):
                try:
                    handle = await asyncio.to_thread(
                        Entrez.efetch,
                        db=database,
                        id=record_ids,
                        rettype="xml",
                        retmode="xml",
                        use_post=True,
                    )
                    records = await asyncio.to_thread(Entrez.read, handle)
                    handle.close()
                    articles = records.get("PubmedArticle", [])
                    books = records.get("PubmedBookArticle", [])
                    fetched_records = list(articles) + list(books)
                    if len(fetched_records) == len(record_ids):
                        logger.debug(
                            "Entrez fetched all %s requested record(s) from '%s'.",
                            len(record_ids),
                            database,
                        )
                    else:
                        logger.warning(
                            "Entrez returned fewer records than requested from '%s' (%s/%s).",
                            database,
                            len(fetched_records),
                            len(record_ids),
                        )
                    if progress_callback:
                        progress_callback(len(record_ids))
                    return fetched_records
                except Exception as e:
                    logger.debug(
                        "Entrez fetch attempt %s/%s failed for database '%s' (%s).",
                        attempt + 1,
                        self.MAX_RETRIES,
                        database,
                        type(e).__name__,
                    )
                    if attempt < self.MAX_RETRIES - 1:
                        await asyncio.sleep(self.RETRY_DELAY)
                    else:
                        logger.warning(
                            "Entrez fetch failed after %s attempts for database '%s'; other retrieval sources may still provide records.",
                            self.MAX_RETRIES,
                            database,
                        )
        return None

    async def _fetch_oai_jats_result(self, pmc_id: str) -> _OaiJatsResult:
        """Fetch OAI JATS once and cache whether its body has usable prose."""
        try:
            xml_text = await self._fetch_oai_jats_for_pmc(pmc_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(
                "PMC OAI-PMH retrieval failed for PMCID %s (%s).",
                pmc_id,
                type(exc).__name__,
            )
            return _OaiJatsResult(None, False)
        if not xml_text:
            return _OaiJatsResult(None, False)

        try:
            body_text = self.pmc_parser.extract_jats_full_text(xml_text)
        except Exception as exc:
            logger.debug(
                "Could not inspect PMC OAI-PMH body for PMCID %s (%s).",
                pmc_id,
                type(exc).__name__,
            )
            return _OaiJatsResult(xml_text, False)
        return _OaiJatsResult(xml_text, self._usable_full_text(body_text))

    async def _cancel_pending_task(self, task: asyncio.Task) -> None:
        """Cancel and drain a retrieval task, closing any response it returned."""
        if not task.done():
            task.cancel()
        try:
            result = await task
            if isinstance(result, _HttpResponse):
                result.close()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def _request_bioc_or_use_oai(
        self,
        pmc_id: str,
        url: str,
        retry_delay: float,
        oai_task: Optional[asyncio.Task[_OaiJatsResult]],
    ) -> tuple[Optional[_HttpResponse], Optional[_OaiJatsResult]]:
        """Race a BioC retry against OAI already started after an HTTP 429."""
        if oai_task is None:
            return await self._request_bioc(url, retry_delay), None

        bioc_task = asyncio.create_task(self._request_bioc(url, retry_delay))
        try:
            done, _ = await asyncio.wait(
                {bioc_task, oai_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if oai_task in done:
                oai_result = await oai_task
                if oai_result.has_usable_body:
                    await self._cancel_pending_task(bioc_task)
                    logger.debug(
                        "PMC OAI-PMH retrieved usable full text for PMCID %s; cancelling the pending BioC retry.",
                        pmc_id,
                    )
                    return None, oai_result

            response = await bioc_task
            if oai_task.done():
                oai_result = await oai_task
                if oai_result.has_usable_body:
                    response.close()
                    logger.debug(
                        "PMC OAI-PMH retrieved usable full text for PMCID %s; using it instead of another BioC retry.",
                        pmc_id,
                    )
                    return None, oai_result
            return response, None
        except BaseException:
            await self._cancel_pending_task(bioc_task)
            raise

    async def _fetch_pmc_full_text_xml(self, pmc_id: str, semaphore: asyncio.Semaphore) -> Optional[str]:
        """Try BioC, OAI-PMH, then Entrez JATS, preserving the endpoint order."""
        id_number = self._normalize_pmc_id(pmc_id)
        url = (
            "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/"
            f"BioC_xml/PMC{id_number}/unicode"
        )
        retryable_statuses = {408, 425, 429}
        oai_task: Optional[asyncio.Task[_OaiJatsResult]] = None

        try:
            for attempt in range(self.BIOC_MAX_RETRIES):
                backoff_delay = min(
                    self.BIOC_INITIAL_RETRY_DELAY * (2 ** attempt),
                    self.BIOC_MAX_RETRY_DELAY,
                )
                response = None
                try:
                    response, oai_result = await self._request_bioc_or_use_oai(
                        pmc_id,
                        url,
                        backoff_delay,
                        oai_task,
                    )
                    if oai_result is not None:
                        return oai_result.xml
                    if response is None:
                        continue
                    if self._is_bioc_no_healthy_upstream(response):
                        break

                    response.raise_for_status()
                    if self._pmc_xml_format(response.text) is None:
                        retry_delay = response.retry_delay or backoff_delay
                        url = response.retry_url or url
                        if attempt < self.BIOC_MAX_RETRIES - 1:
                            logger.warning(
                                "PMC BioC returned a non-XML response for PMCID '%s' on attempt %s/%s; retrying in %.0f seconds.",
                                pmc_id,
                                attempt + 1,
                                self.BIOC_MAX_RETRIES,
                                retry_delay,
                            )
                            # _request_bioc owns the shared cooldown; the next
                            # request waits there, avoiding a second sleep here.
                            continue
                        break

                    logger.debug("PMC BioC returned parseable XML for PMCID '%s'.", pmc_id)
                    return response.text
                except requests.RequestException as exc:
                    error_response = getattr(exc, "response", None)
                    status_code = (
                        error_response.status_code
                        if error_response is not None
                        else None
                    )
                    is_retryable = (
                        status_code is None
                        or status_code in retryable_statuses
                        or (status_code is not None and 500 <= status_code <= 599)
                    )
                    if status_code == 429 and oai_task is None:
                        logger.debug(
                            "PMC BioC rate-limited PMCID '%s'; starting OAI-PMH retrieval while waiting to retry BioC.",
                            pmc_id,
                        )
                        oai_task = asyncio.create_task(
                            self._fetch_oai_jats_result(pmc_id)
                        )
                    if not is_retryable:
                        break
                    if attempt >= self.BIOC_MAX_RETRIES - 1:
                        break

                    retry_delay = (
                        error_response.retry_delay
                        if error_response is not None and error_response.retry_delay is not None
                        else backoff_delay
                    )
                    if error_response is not None:
                        url = error_response.retry_url or url
                    logger.warning(
                        "PMC BioC is temporarily unavailable for PMCID '%s' on attempt %s/%s (HTTP %s); retrying in %.0f seconds.",
                        pmc_id,
                        attempt + 1,
                        self.BIOC_MAX_RETRIES,
                        status_code or "network error",
                        retry_delay,
                    )
                finally:
                    if response is not None:
                        response.close()

            return await self._fetch_jats_fallback_xml(
                pmc_id,
                semaphore,
                oai_task,
            )
        finally:
            if oai_task is not None:
                await self._cancel_pending_task(oai_task)

    async def _request_bioc(self, url: str, retry_delay: float) -> _HttpResponse:
        """Serialize BioC traffic and honor server cooldowns between attempts."""
        async with self._bioc_request_lock:
            wait_seconds = self._bioc_next_request_at - time.monotonic()
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            try:
                response = await asyncio.to_thread(self._get_url_response, url, timeout=30)
            except requests.RequestException:
                self._bioc_next_request_at = max(
                    self._bioc_next_request_at,
                    time.monotonic() + retry_delay,
                )
                raise

            response_time = time.monotonic()
            needs_retry = (
                response.status_code in {408, 425, 429}
                or response.status_code >= 500
                or self._pmc_xml_format(response.text) is None
            )
            if needs_retry:
                response.retry_delay, response.retry_url = self._bioc_retry_info(
                    response, url, retry_delay
                )
                pause = response.retry_delay
            else:
                pause = self.BIOC_MIN_REQUEST_INTERVAL
            self._bioc_next_request_at = max(
                self._bioc_next_request_at,
                response_time + pause,
            )
            return response

    async def _fetch_oai_jats_for_pmc(self, pmc_id: str) -> Optional[str]:
        """Retrieve one reusable PMC article record through OAI-PMH as JATS XML."""
        identifier = f"oai:pubmedcentral.nih.gov:{self._normalize_pmc_id(pmc_id)}"
        params = {
            "verb": "GetRecord",
            "identifier": identifier,
            "metadataPrefix": "pmc",
        }
        retryable_statuses = {408, 425, 429}
        terminal_oai_errors = {
            "badArgument",
            "badVerb",
            "cannotDisseminateFormat",
            "idDoesNotExist",
            "noRecordsMatch",
            "noMetadataFormats",
        }

        async def retry_after_failure(
            attempt: int,
            delay: float,
            failure: str,
        ) -> bool:
            if attempt >= self.PMC_OAI_MAX_RETRIES - 1:
                logger.debug(
                    "PMC OAI-PMH failed for PMCID %s after %s attempts (%s).",
                    pmc_id,
                    self.PMC_OAI_MAX_RETRIES,
                    failure,
                )
                return False

            wait_until = time.monotonic() + max(
                self.PMC_OAI_MIN_REQUEST_INTERVAL,
                delay,
            )
            self._pmc_oai_next_request_at = max(
                self._pmc_oai_next_request_at,
                wait_until,
            )
            wait_seconds = max(
                0.0,
                self._pmc_oai_next_request_at - time.monotonic(),
            )
            logger.debug(
                "PMC OAI-PMH attempt %s/%s failed for PMCID %s (%s); retrying in %.0f seconds.",
                attempt + 1,
                self.PMC_OAI_MAX_RETRIES,
                pmc_id,
                failure,
                wait_seconds,
            )
            await asyncio.sleep(wait_seconds)
            return True

        # PMC requires OAI-PMH calls to be serial. The interval also keeps the
        # request rate below the documented three requests per second limit.
        async with self._pmc_oai_request_lock:
            for attempt in range(self.PMC_OAI_MAX_RETRIES):
                wait_seconds = self._pmc_oai_next_request_at - time.monotonic()
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)
                response = None
                backoff_delay = min(
                    self.PMC_OAI_INITIAL_RETRY_DELAY * (2 ** attempt),
                    self.PMC_OAI_MAX_RETRY_DELAY,
                )
                try:
                    request_url = f"{self.PMC_OAI_BASE_URL}?{urlencode(params, safe=':')}"
                    response = await asyncio.to_thread(
                        self._get_url_response,
                        request_url,
                        timeout=30,
                    )
                    response_time = time.monotonic()
                    self._pmc_oai_next_request_at = (
                        response_time + self.PMC_OAI_MIN_REQUEST_INTERVAL
                    )
                    xml_text = response.text
                    try:
                        root = ET.fromstring(xml_text)
                    except ET.ParseError:
                        content_type = response.headers.get(
                            "Content-Type", "unknown"
                        ).split(";", 1)[0]
                        is_retryable = (
                            response.status_code < 400
                            or response.status_code in retryable_statuses
                            or response.status_code >= 500
                        )
                        if is_retryable:
                            retry_delay = max(
                                backoff_delay,
                                self._parse_retry_after(
                                    response.headers.get("Retry-After")
                                ) or 0,
                            )
                            if await retry_after_failure(
                                attempt,
                                retry_delay,
                                f"HTTP {response.status_code}, non-XML {content_type}",
                            ):
                                continue
                            return None
                        logger.debug(
                            "PMC OAI-PMH returned a non-XML response for PMCID %s "
                            "(HTTP %s, Content-Type %s).",
                            pmc_id,
                            response.status_code,
                            content_type,
                        )
                        return None

                    if self._local_name(root).lower() != "oai-pmh":
                        is_retryable = (
                            response.status_code < 400
                            or response.status_code in retryable_statuses
                            or response.status_code >= 500
                        )
                        if is_retryable and await retry_after_failure(
                            attempt,
                            backoff_delay,
                            f"HTTP {response.status_code}, unexpected XML",
                        ):
                            continue
                        if is_retryable:
                            return None
                        logger.debug(
                            "PMC OAI-PMH returned an unexpected XML document for PMCID %s "
                            "(HTTP %s).",
                            pmc_id,
                            response.status_code,
                        )
                        return None

                    error = next(
                        (
                            node
                            for node in root.iter()
                            if self._local_name(node).lower() == "error"
                        ),
                        None,
                    )
                    if error is not None:
                        error_code = error.attrib.get("code", "unknown")
                        if error_code == "cannotDisseminateFormat":
                            return None
                        if error_code in terminal_oai_errors:
                            logger.debug(
                                "PMC OAI-PMH returned terminal error '%s' for PMCID %s.",
                                error_code,
                                pmc_id,
                            )
                            return None
                        retry_delay = max(
                            backoff_delay,
                            self._parse_retry_after(
                                response.headers.get("Retry-After")
                            ) or 0,
                        )
                        if await retry_after_failure(
                            attempt,
                            retry_delay,
                            f"OAI error {error_code}",
                        ):
                            continue
                        return None

                    is_retryable_status = (
                        response.status_code in retryable_statuses
                        or response.status_code >= 500
                    )
                    if is_retryable_status:
                        retry_delay = max(
                            backoff_delay,
                            self._parse_retry_after(
                                response.headers.get("Retry-After")
                            ) or 0,
                        )
                        if await retry_after_failure(
                            attempt,
                            retry_delay,
                            f"HTTP {response.status_code}",
                        ):
                            continue
                        return None
                    if not response.ok:
                        logger.debug(
                            "PMC OAI-PMH returned HTTP %s for PMCID %s.",
                            response.status_code,
                            pmc_id,
                        )
                        return None

                    names = {self._local_name(node).lower() for node in root.iter()}
                    if "record" not in names or "article" not in names or "body" not in names:
                        logger.debug(
                            "PMC OAI-PMH did not return a full-text JATS body for PMCID %s.",
                            pmc_id,
                        )
                        return None
                    if self._pmc_xml_format(xml_text) != "jats":
                        logger.debug(
                            "PMC OAI-PMH response was not parseable JATS for PMCID %s.",
                            pmc_id,
                        )
                        return None

                    logger.debug("PMC OAI-PMH returned JATS full text for PMCID %s.", pmc_id)
                    return xml_text
                except requests.RequestException as e:
                    if await retry_after_failure(
                        attempt,
                        backoff_delay,
                        type(e).__name__,
                    ):
                        continue
                    return None
                except Exception as e:
                    logger.debug(
                        "PMC OAI-PMH processing failed for PMCID %s (%s).",
                        pmc_id,
                        type(e).__name__,
                    )
                    return None
                finally:
                    if response is not None:
                        response.close()

        return None

    @staticmethod
    def _parse_retry_after(value: Optional[str]) -> Optional[float]:
        """Parse Retry-After as delay-seconds or an HTTP date."""
        if not value:
            return None
        try:
            return max(0.0, float(value.strip()))
        except (TypeError, ValueError):
            pass
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())

    async def _fetch_entrez_jats_for_pmc(self, pmc_id: str, semaphore: asyncio.Semaphore) -> Optional[str]:
        """Fetch standard PMC JATS XML through Entrez EFetch."""
        id_number = pmc_id.upper().replace("PMC", "")
        async with semaphore:
            for attempt in range(self.MAX_RETRIES):
                handle = None
                try:
                    handle = await asyncio.to_thread(
                        Entrez.efetch,
                        db="pmc",
                        id=id_number,
                        retmode="xml",
                    )
                    raw_xml = await asyncio.to_thread(handle.read)
                    if isinstance(raw_xml, bytes):
                        raw_xml = raw_xml.decode("utf-8", errors="replace")
                    raw_xml = str(raw_xml)
                    if self._pmc_xml_format(raw_xml) == "jats":
                        logger.debug("Entrez returned parseable JATS XML for PMCID '%s'.", pmc_id)
                        return raw_xml
                    raise ValueError("Entrez PMC response was not parseable JATS XML.")
                except Exception as e:
                    if attempt < self.MAX_RETRIES - 1:
                        logger.debug(
                            "PMC JATS attempt %s/%s failed for %s (%s); retrying.",
                            attempt + 1,
                            self.MAX_RETRIES,
                            pmc_id,
                            type(e).__name__,
                        )
                        await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
                    else:
                        logger.warning(
                            "PMC JATS retrieval failed for %s after %s attempts (%s).",
                            pmc_id,
                            self.MAX_RETRIES,
                            type(e).__name__,
                        )
                finally:
                    if handle is not None:
                        handle.close()
        return None
