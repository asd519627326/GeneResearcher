# src/core/parsers.py

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import xml.etree.ElementTree as ET
import re
from .data_models import Paper 

logger = logging.getLogger(__name__)


class PubMedXMLParser:
    # ... (other methods of the class if any) ...
    @staticmethod
    def _parse_date_element(value: Any) -> Optional[datetime]:
        """Parse common PubMed date elements, including ``MedlineDate``."""
        if not value or not hasattr(value, "get"):
            return None

        year = value.get("Year")
        month = value.get("Month", "1")
        day = value.get("Day", "1")

        if not year:
            medline_date = str(value.get("MedlineDate", "")).strip()
            match = re.search(
                r"(?P<year>\d{4})(?:\s+(?P<month>[A-Za-z]{3,9}))?",
                medline_date,
            )
            if not match:
                return None
            year = match.group("year")
            month = match.group("month") or "1"

        try:
            month_text = str(month)
            if not month_text.isdigit():
                month_text = month_text[:3].title()
                month = datetime.strptime(month_text, "%b").month
            return datetime(int(year), int(month), int(day))
        except (TypeError, ValueError):
            return None

    def _extract_publication_date(
        self,
        article: Dict[str, Any],
        pubmed_data: Dict[str, Any],
    ) -> Optional[datetime]:
        """Return the earliest public publication date represented by PubMed.

        Electronic ``ArticleDate`` is considered alongside the journal issue
        date and PubMed/Entrez indexing dates.  Received and accepted dates are
        deliberately ignored because they do not make a paper available to a
        time-restricted literature search.
        """
        candidates = []

        article_dates = article.get("ArticleDate", [])
        if article_dates and not isinstance(article_dates, list):
            article_dates = [article_dates]
        for date_value in article_dates or []:
            parsed = self._parse_date_element(date_value)
            if parsed:
                candidates.append(parsed)

        journal_date = article.get("Journal", {}).get(
            "JournalIssue", {}
        ).get("PubDate", {})
        parsed_journal_date = self._parse_date_element(journal_date)
        if parsed_journal_date:
            candidates.append(parsed_journal_date)

        history = pubmed_data.get("History", []) if pubmed_data else []
        if history and not isinstance(history, list):
            history = [history]
        for date_value in history or []:
            attributes = getattr(date_value, "attributes", {})
            status = str(attributes.get("PubStatus", "")).lower()
            if status not in {"pubmed", "entrez"}:
                continue
            parsed = self._parse_date_element(date_value)
            if parsed:
                candidates.append(parsed)

        return min(candidates) if candidates else None

    def parse(self, pubmed_records: Any, require_abstract: bool = True) -> List[Paper]:
        """
        解析 Bio.Entrez 返回的记录列表，支持 PubmedArticle 和 PubmedBookArticle。
        """
        papers = []
        
        # 兼容性处理：如果传入的是字典（通常包含 'PubmedArticle' 和 'PubmedBookArticle' 列表）
        # 或者是列表
        all_articles = []
        
        if isinstance(pubmed_records, dict):
            # 提取期刊文章
            if 'PubmedArticle' in pubmed_records:
                all_articles.extend([('journal', art) for art in pubmed_records['PubmedArticle']])
            # 提取书籍/GeneReviews文章
            if 'PubmedBookArticle' in pubmed_records:
                all_articles.extend([('book', art) for art in pubmed_records['PubmedBookArticle']])
        elif isinstance(pubmed_records, list):
            # 假设列表里混杂了字典，或者只是普通文章列表
            for item in pubmed_records:
                if 'BookDocument' in item:
                    all_articles.append(('book', item))
                else:
                    all_articles.append(('journal', item))

        if not all_articles:
            return papers

        for art_type, article_data in all_articles:
            try:
                extracted_info = None
                if art_type == 'journal':
                    extracted_info = self._extract_article_info(article_data, require_abstract)
                elif art_type == 'book':
                    extracted_info = self._extract_book_info(article_data, require_abstract)
            
                if extracted_info:
                    paper_obj = self._create_paper_object(extracted_info)
                    if paper_obj: 
                        papers.append(paper_obj)
            except Exception as e:
                # 错误处理逻辑
                pmid_for_log = 'UNKNOWN'
                try:
                    if art_type == 'journal':
                        pmid_for_log = str(article_data.get('MedlineCitation', {}).get('PMID', 'UNKNOWN'))
                    else:
                        pmid_for_log = str(article_data.get('BookDocument', {}).get('PMID', 'UNKNOWN'))
                except Exception: pass
                logger.error(
                    "Could not parse PubMed record %s (%s).",
                    pmid_for_log,
                    type(e).__name__,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
        
        return papers

    
    def _create_paper_object(self, info: Dict[str, Any]) -> Optional[Paper]:
        # [FINAL] Ensures pmid key exists before creating a Paper object
        pmid = info.get('pmid')
        if not pmid:
            logger.warning("Skipping a PubMed record because parsed metadata has no PMID.")
            return None
        return Paper(
            paper_id=pmid,
            title=info.get('title', 'No Title Found'),
            abstract=info.get('abstract', 'No abstract found.'),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            published_date=info.get('published_date_obj'),
            updated_date=info.get('published_date_obj'),
            source='pubmed',
            keywords=info.get('keywords_list', []),
            doi=info.get('doi', ''),
            full_text=None,
            authors=info.get('authors', []),
            publication_types=info.get('publication_types', [])
        )


    def _extract_book_info(self, article_data: Dict[str, Any], require_abstract: bool = True) -> Optional[Dict[str, Any]]:
        """
        [新增] 专门解析 PubmedBookArticle (如 GeneReviews)
        """
        info = {}
        book_doc = article_data.get('BookDocument', {})
        if not book_doc: return None

        # 1. PMID
        pmid = book_doc.get('PMID')
        if not pmid: return None
        info['pmid'] = str(pmid)

        # 2. Title
        info['title'] = str(book_doc.get('ArticleTitle', ''))

        # 3. Abstract (Book摘要通常是结构化的，非常重要)
        abstract_data = book_doc.get('Abstract', {}).get('AbstractText', [])
        processed_abstracts = []
        if isinstance(abstract_data, list):
            for item in abstract_data:
                text = str(item)
                # 提取 Label (如 "CLINICAL CHARACTERISTICS")
                if hasattr(item, 'attributes') and 'Label' in item.attributes:
                    label = item.attributes['Label']
                    processed_abstracts.append(f"**{label}**: {text}")
                else:
                    processed_abstracts.append(text)
        elif isinstance(abstract_data, str):
            processed_abstracts.append(abstract_data)
        
        abstract_text = '\n\n'.join(processed_abstracts).strip()
        if require_abstract and not abstract_text:
            return None
        info['abstract'] = abstract_text

        # 4. Authors (BookDocument 下的 AuthorList 通常是章节作者，Book 下的是编辑)
        authors = []
        author_list = book_doc.get('AuthorList', [])
        # 有时它是列表的列表，有时直接是列表
        if author_list and isinstance(author_list, list):
            # Biopython 解析出来有时是 ListElement
            for auth_group in author_list:
                # 只要 Type='authors' 的
                if hasattr(auth_group, 'attributes') and auth_group.attributes.get('Type') == 'authors':
                    for author in auth_group:
                        if hasattr(author, 'get'):
                            last = author.get('LastName', '')
                            initials = author.get('Initials', '')
                            if last:
                                authors.append(f"{last} {initials}".strip())
        info['authors'] = authors

        # 5. Date (优先使用 DateRevised，因为 Books 经常更新)
        date_revised = book_doc.get('DateRevised')
        date_contrib = book_doc.get('ContributionDate')
        
        target_date = date_revised if date_revised else date_contrib
        pub_date_obj = None
        if target_date:
            try:
                year = target_date.get('Year')
                month = target_date.get('Month', '1')
                day = target_date.get('Day', '1')
                pub_date_obj = datetime(int(year), int(month), int(day))
            except (TypeError, ValueError, OverflowError) as exc:
                logger.warning(
                    "Could not parse the publication date for PMID %s (%s).",
                    info.get("pmid", "unknown"),
                    type(exc).__name__,
                )
        info['published_date_obj'] = pub_date_obj

        # 6. Keywords
        keywords = []
        kw_list = book_doc.get('KeywordList', [])
        for kw_group in kw_list:
            for kw in kw_group:
                keywords.append(str(kw))
        info['keywords_list'] = keywords

        # 7. Types
        info['publication_types'] = ['Book', 'Review'] # 默认为 Book Review

        # 8. DOI (通常在 ArticleIdList 中)
        doi = ''
        id_list = book_doc.get('ArticleIdList', [])
        for aid in id_list:
            if hasattr(aid, 'attributes') and aid.attributes.get('IdType') == 'doi':
                doi = str(aid)
                break
        info['doi'] = doi

        return info



    
    def _extract_article_info(self, article_data: Dict[str, Any], require_abstract: bool = True) -> Optional[Dict[str, Any]]:
        try:
            info = {}
            medline_citation = article_data.get('MedlineCitation', {})
            pubmed_data = article_data.get('PubmedData', {}) 

            if not medline_citation: return None
            
            pmid = medline_citation.get('PMID')
            if not pmid: return None
            info['pmid'] = str(pmid)

            article = medline_citation.get('Article', {})
            info['title'] = str(article.get('ArticleTitle', ''))

            # --- [修改] 使用更健壮的逻辑提取作者列表 ---
            try:
                authors = []
                author_list_raw = article.get('AuthorList')

                if author_list_raw:
                    # [新增] 确保我们迭代的是一个列表，以防只有一个作者时API返回单个对象
                    if not isinstance(author_list_raw, list):
                        author_list_raw = [author_list_raw]
                    
                    for author in author_list_raw:
                        # 确保author条目是可解析的
                        if not hasattr(author, 'get'):
                            continue

                        # [新增] 优先处理团体作者
                        collective_name = author.get('CollectiveName')
                        if collective_name:
                            authors.append(str(collective_name).strip())
                            continue  # 处理完团体作者后，跳到下一个条目

                        # 处理个人作者
                        last_name = author.get('LastName')
                        initials = author.get('Initials')
                        
                        # 确保我们有姓氏，并且它是字符串
                        if last_name and isinstance(last_name, str):
                            author_str = f"{last_name} {initials}".strip() if initials and isinstance(initials, str) else last_name
                            authors.append(author_str)
                
                info['authors'] = authors

            except Exception as e:
                # [新增] 错误隔离：如果作者解析失败，记录警告并继续，而不是让整个文章解析失败
                logger.warning("Could not parse authors for PMID %s (%s).", pmid, type(e).__name__)
                info['authors'] = []  # 确保字段存在但为空列表

            # --- [新增] 提取文章类型列表 ---
            publication_types = []
            if 'PublicationTypeList' in article:
                try:
                    pub_type_list_raw = article['PublicationTypeList']
                    # 统一处理单个对象和列表的情况
                    if not isinstance(pub_type_list_raw, list):
                        pub_type_list_raw = [pub_type_list_raw]
                    # 安全地将每个元素(Bio.Entrez.StringElement)转换为字符串
                    publication_types = [str(ptype) for ptype in pub_type_list_raw]
                except (KeyError, TypeError, AttributeError) as e:
                    logger.warning(
                        "Could not parse publication types for PMID %s (%s).",
                        pmid,
                        type(e).__name__,
                    )
            info['publication_types'] = publication_types
            
            # --- 其他字段提取逻辑 (无变化) ---
            abstract_section = article.get('Abstract', {})
            abstract_texts_raw = abstract_section.get('AbstractText', [])
            processed_abstracts = []
            if isinstance(abstract_texts_raw, list):
                for item in abstract_texts_raw:
                    if isinstance(item, str): processed_abstracts.append(item)
                    elif hasattr(item, 'attributes') and 'Label' in item.attributes: processed_abstracts.append(f"**{item.attributes['Label']}**: {str(item)}")
                    else: processed_abstracts.append(str(item))
            elif isinstance(abstract_texts_raw, str): processed_abstracts.append(abstract_texts_raw)
            abstract_text = '\n\n'.join(processed_abstracts).strip()
            if require_abstract and not abstract_text:
                logger.warning("Skipping PubMed record %s because it has no abstract.", info["pmid"])
                return None 
            info['abstract'] = abstract_text

            info['published_date_obj'] = self._extract_publication_date(
                article,
                pubmed_data,
            )

            keywords = []
            mesh_headings = medline_citation.get('MeshHeadingList', [])
            if mesh_headings:
                for heading in mesh_headings:
                    if 'DescriptorName' in heading: keywords.append(str(heading['DescriptorName']))
            keyword_list_outer = medline_citation.get('KeywordList', [])
            if keyword_list_outer:
                for keyword_group in keyword_list_outer:
                    if isinstance(keyword_group, list): keywords.extend([str(kw) for kw in keyword_group if isinstance(kw, str) or hasattr(kw, 'attributes')])
            info['keywords_list'] = sorted(list(set(keywords)))

            doi = ''
            if pubmed_data and 'ArticleIdList' in pubmed_data:
                for aid in pubmed_data['ArticleIdList']:
                    if hasattr(aid, 'attributes') and aid.attributes.get('IdType') == 'doi':
                        doi = str(aid)
                        break
            if not doi and 'ELocationID' in article:
                for eid in article['ELocationID']:
                     if hasattr(eid, 'attributes') and eid.attributes.get('EIdType') == 'doi':
                        doi = str(eid)
                        break
            info['doi'] = doi

            return info
        except Exception as e:
            pmid_for_log = article_data.get('MedlineCitation', {}).get('PMID', 'UNKNOWN')
            logger.error(
                "Could not parse PubMed XML record %s (%s).",
                pmid_for_log,
                type(e).__name__,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return None


class PMCBioCParser:
    """
    一个高度专业化的解析器，其唯一职责是从PMC BioC XML中提取格式化的全文。
    """
    
    def __init__(self, entrez_email: str,max_text_length: int = 120000):
        self.entrez_email = entrez_email
        self.max_text_length = max_text_length  # 保存阈值
        self.EXCLUDED_SECTION_TYPES = { "title", "abstract", "author", "kwd", "ref", "fig", "table", "ack_fund", "materials and methods", "experimental procedures", "abbreviations", "keywords", "figure", "supplementary material", "supporting information", "ack", "acknowledgments", "funding", "author contributions", "author_cont", "comp_int", "references", "ethics statement", "data availability", "suppl", "abbr", "auth_cont", "suppl", "figures", "tables", "appendix", "review_info" }
        self.REVIEW_PUBLICATION_TYPES = { "review", "systematic review", "meta-analysis" }
        
        # [新增] 定义并编译用于过滤样板句的正则表达式
        self.BOILERPLATE_PATTERNS = [
            re.compile(r"^\s*not applicable\s*\.?\s*$", re.IGNORECASE),
            re.compile(r"^\s*all authors (?:have )?(?:read and )?(?:approved|agreed|consent)", re.IGNORECASE),
            re.compile(r"^\s*this is not applicable", re.IGNORECASE),
            re.compile(r"^\s*(?:the )?(?:data|material|code) (?:that support|supporting) the (?:findings|conclusion)", re.IGNORECASE),
            re.compile(r"^\s*availability of data and materials\s*$", re.IGNORECASE),
            re.compile(r"^\s*competing interests?\s*$", re.IGNORECASE),
            re.compile(r"^\s*author contributions?\s*$", re.IGNORECASE),
            re.compile(r"^\s*ethics approval(?: and consent to participate)?\s*$", re.IGNORECASE),
            re.compile(r"^\s*consent for publication\s*$", re.IGNORECASE)
        ]

    # [修改] 简化决策逻辑并增加内容过滤
    def extract_full_text(self, xml_text: str, publication_types: Optional[List[str]] = None) -> str:
        if not xml_text:
            return "No structured body prose found."

        try:
            root = ET.fromstring(xml_text)
            doc_el = root.find('.//document')
            if doc_el is None:
                return "No structured body prose found."

            pmcid = doc_el.findtext('./id', 'UNKNOWN')
            

            is_review_article = False
            if publication_types:
                types_lower = {pt.lower() for pt in publication_types}
                if not self.REVIEW_PUBLICATION_TYPES.isdisjoint(types_lower):
                    is_review_article = True

            # --- 文本收集 ---
            full_text_parts = []
            for passage_el in doc_el.findall('./passage'):
                infons = {infon.get('key'): "".join(infon.itertext()).strip() for infon in passage_el.findall('./infon')}
                passage_text = "".join(passage_el.find('./text').itertext()).strip() if passage_el.find('./text') is not None else ""
                
                section_type = infons.get('section_type', '').lower()

                # 根据最终的 is_review_article 标志位来决定是否跳过
                is_methods_section = 'method' in section_type
                if is_methods_section and not is_review_article:
                    logger.debug("Skipping the Methods section for non-review article PMCID %s.", pmcid)
                    continue

                if section_type and section_type not in self.EXCLUDED_SECTION_TYPES and passage_text:
                    passage_type = infons.get('type', '').lower()
                    if passage_type.startswith('title_'):
                        try:
                            level = int(passage_type.split('_')[-1])
                            full_text_parts.append(f"\n{'#' * (level + 1)} {passage_text}\n")
                        except ValueError:
                            full_text_parts.append(f"\n### {passage_text}\n")
                    elif passage_type == 'paragraph':
                        full_text_parts.append(passage_text)
            
            # --- [新增] 内容过滤步骤 ---
            filtered_parts = []
            for part in full_text_parts:
                is_boilerplate = False
                # 只对段落进行检查，跳过标题
                if not part.strip().startswith('#'):
                    for pattern in self.BOILERPLATE_PATTERNS:
                        if pattern.search(part):
                            is_boilerplate = True
                            logger.debug("Filtered boilerplate text from PMCID %s.", pmcid)
                            break
                if not is_boilerplate:
                    filtered_parts.append(part)

            final_text = "\n\n".join(filtered_parts).strip()
            
            # 如果文本过长，返回 None
            if len(final_text) > self.max_text_length:
                logger.warning(
                    "Discarded full text for PMCID %s because it exceeds the %s-character limit (%s characters).",
                    pmcid,
                    self.max_text_length,
                    len(final_text),
                )
                return None
            
            return final_text or "No structured body prose found."

        except ET.ParseError as e:
            logger.warning("Could not parse PMC BioC XML to extract full text (%s).", type(e).__name__)
            return "Failed to parse XML for full text."

    def extract_jats_full_text(
        self,
        xml_text: str,
        publication_types: Optional[List[str]] = None,
    ) -> str:
        """Extract body prose from JATS XML returned by PMC OAI-PMH or Entrez."""
        if not xml_text:
            return "No structured body prose found."

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.warning("Could not parse PMC JATS XML (%s).", type(exc).__name__)
            return "Failed to parse XML for full text."

        def local_name(element: ET.Element) -> str:
            return str(element.tag).rsplit("}", 1)[-1].lower()

        article = next((node for node in root.iter() if local_name(node) == "article"), None)
        body = next(
            (node for node in article if local_name(node) == "body"),
            None,
        ) if article is not None else None
        if body is None:
            return "No structured body prose found."

        types_lower = {str(value).lower() for value in (publication_types or [])}
        is_review_article = not self.REVIEW_PUBLICATION_TYPES.isdisjoint(types_lower)
        excluded_titles = {
            "abbreviations",
            "acknowledgments",
            "acknowledgements",
            "appendix",
            "author contributions",
            "competing interests",
            "conflict of interest",
            "consent for publication",
            "data availability",
            "ethics statement",
            "funding",
            "references",
            "supplementary material",
        }
        text_parts = []

        def clean_element_text(element: Optional[ET.Element]) -> str:
            if element is None:
                return ""
            return re.sub(r"\s+", " ", "".join(element.itertext())).strip()

        def is_boilerplate(text: str) -> bool:
            return any(pattern.search(text) for pattern in self.BOILERPLATE_PATTERNS)

        def walk(container: ET.Element, inherited_exclusion: bool = False) -> None:
            title_element = next(
                (child for child in container if local_name(child) == "title"),
                None,
            )
            title = clean_element_text(title_element)
            section_type = str(container.attrib.get("sec-type", "")).lower()
            normalized_title = title.lower().rstrip(":")
            excluded = inherited_exclusion or section_type in self.EXCLUDED_SECTION_TYPES
            excluded = excluded or normalized_title in excluded_titles
            if not is_review_article and (
                "method" in section_type or normalized_title.startswith("method")
            ):
                excluded = True

            if title and not excluded:
                text_parts.append(f"### {title}")

            for child in container:
                tag = local_name(child)
                if tag == "sec":
                    walk(child, excluded)
                elif tag == "p" and not excluded:
                    paragraph = clean_element_text(child)
                    if paragraph and not is_boilerplate(paragraph):
                        text_parts.append(paragraph)

        walk(body)
        final_text = "\n\n".join(text_parts).strip()
        if len(final_text) > self.max_text_length:
            logger.warning(
                "PMC JATS document dropped: Text length (%s) exceeds limit (%s).",
                len(final_text),
                self.max_text_length,
            )
            return None
        return final_text or "No structured body prose found."
