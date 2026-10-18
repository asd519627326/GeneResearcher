import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from generesearcher.connectors.ncbi_connector import NcbiApiConnector, _HttpResponse


class BioCNoHealthyUpstreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_healthy_upstream_skips_remaining_bioc_retries(self):
        oai_jats = "<article><body><p>Article prose.</p></body></article>"

        for status_code in (429, 503):
            with self.subTest(status_code=status_code):
                connector = NcbiApiConnector.__new__(NcbiApiConnector)
                connector._bioc_next_request_at = 0.0
                connector.pmc_parser = SimpleNamespace(
                    extract_jats_full_text=lambda xml: "Article prose." if xml == oai_jats else ""
                )
                response = _HttpResponse(
                    status_code=status_code,
                    headers={"Content-Type": "text/plain"},
                    content=b"no healthy upstream",
                )
                connector._request_bioc = AsyncMock(return_value=response)
                connector._fetch_oai_jats_for_pmc = AsyncMock(return_value=oai_jats)
                connector._fetch_entrez_jats_for_pmc = AsyncMock(return_value=None)

                result = await connector._fetch_pmc_full_text_xml(
                    "PMC123",
                    asyncio.Semaphore(1),
                )

                self.assertEqual(result, oai_jats)
                connector._request_bioc.assert_awaited_once()
                connector._fetch_oai_jats_for_pmc.assert_awaited_once_with("PMC123")
                connector._fetch_entrez_jats_for_pmc.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
