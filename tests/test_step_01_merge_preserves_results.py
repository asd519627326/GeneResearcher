import asyncio
import unittest

from generesearcher.connectors.ncbi_connector import PaperFetchResult
from generesearcher.core.data_models import Paper
from generesearcher.pipeline.step_01_retrieval import _retrieve_and_parse_papers


class _FakeNcbiConnector:
    def __init__(self, pubmed_papers, pmc_papers):
        self.pubmed_papers = pubmed_papers
        self.pmc_papers = pmc_papers

    async def search_ids(self, query, database, max_results, semaphore):
        if database == "pubmed":
            return [paper.paper_id for paper in self.pubmed_papers[:max_results]]
        return [f"PMC{paper.paper_id}" for paper in self.pmc_papers[:max_results]]

    async def fetch_papers_by_pubmed_ids(self, pmids, semaphore, progress_callback=None):
        if progress_callback:
            progress_callback(len(pmids))
        return PaperFetchResult(self.pubmed_papers)

    async def fetch_papers_by_pmc_ids(self, pmc_ids, semaphore, progress_callback=None):
        if progress_callback:
            progress_callback(len(pmc_ids))
        source_id_to_paper_id = {
            paper.paper_id: paper.paper_id for paper in self.pmc_papers
        }
        return PaperFetchResult(self.pmc_papers, source_id_to_paper_id)


class StepOneMergePreservesResultsTests(unittest.IsolatedAsyncioTestCase):
    async def _retrieve(self, pubmed_papers, pmc_papers, limit=2):
        return await _retrieve_and_parse_papers(
            connector=_FakeNcbiConnector(pubmed_papers, pmc_papers),
            semaphore=asyncio.Semaphore(3),
            gene="GENE",
            function_name="function",
            keywords=[],
            max_papers_per_gene=limit,
            exclude_pmids=[],
            publication_date_before=None,
            gene_aliases=None,
            show_progress=False,
        )

    async def test_all_unique_pubmed_and_pmc_articles_are_retained(self):
        pubmed_papers = [
            Paper(paper_id="101", title="PubMed one", abstract="Abstract one", source="pubmed"),
            Paper(paper_id="102", title="PubMed two", abstract="Abstract two", source="pubmed"),
        ]
        pmc_papers = [
            Paper(
                paper_id="300",
                title="PMC full text",
                abstract="Abstract three",
                full_text="Full-text body",
                source="pmc_oai",
            )
        ]

        papers = await self._retrieve(pubmed_papers, pmc_papers)

        self.assertEqual([paper.paper_id for paper in papers], ["101", "102", "300"])
        self.assertTrue(papers[2].full_text)

    async def test_duplicate_is_enriched_and_other_unique_articles_are_retained(self):
        pubmed_papers = [
            Paper(paper_id="101", title="PubMed one", abstract="Abstract one", source="pubmed"),
            Paper(paper_id="102", title="PubMed two", abstract="Abstract two", source="pubmed"),
        ]
        pmc_papers = [
            Paper(paper_id="101", title="PMC one", full_text="Full-text one", source="pmc_oai"),
            Paper(paper_id="300", title="PMC two", full_text="Full-text two", source="pmc_oai"),
        ]

        papers = await self._retrieve(pubmed_papers, pmc_papers)

        self.assertEqual([paper.paper_id for paper in papers], ["101", "102", "300"])
        self.assertTrue(papers[0].full_text)
        self.assertEqual(papers[0].abstract, "Abstract one")

    async def test_article_without_usable_full_text_is_still_retained(self):
        pubmed_papers = [
            Paper(paper_id="101", title="PubMed one", abstract="Abstract one", source="pubmed"),
            Paper(paper_id="102", title="PubMed two", abstract="Abstract two", source="pubmed"),
        ]
        pmc_papers = [
            Paper(
                paper_id="300",
                title="PMC without body",
                full_text="No structured body prose found.",
                source="pmc_oai",
            )
        ]

        papers = await self._retrieve(pubmed_papers, pmc_papers)

        self.assertEqual([paper.paper_id for paper in papers], ["101", "102", "300"])


if __name__ == "__main__":
    unittest.main()
