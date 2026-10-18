import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from generesearcher import AnalysisResult, AnalysisSession, ResearchConfig


def _config(project_folder: str) -> ResearchConfig:
    return ResearchConfig(
        genes=["TP53"],
        function_name="test function",
        research_background_template="Study of {gene} in {function_name}",
        project_folder=project_folder,
    )


class AnalysisSessionTests(unittest.IsolatedAsyncioTestCase):
    def _make_session(self, runner):
        model_loader = patch(
            "generesearcher.runtime_dependencies.load_scispacy_model",
            return_value=object(),
        )
        runner_factory = patch("generesearcher.run._PipelineExecutor", return_value=runner)
        model_loader.start()
        runner_factory.start()
        self.addCleanup(model_loader.stop)
        self.addCleanup(runner_factory.stop)
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        session = AnalysisSession(_config(workspace.name))
        self.addCleanup(session.close)
        return session

    async def test_steps_must_be_run_in_order_without_preparing_database(self):
        runner = Mock()
        session = self._make_session(runner)

        with self.assertRaisesRegex(RuntimeError, "next stage is 'literature retrieval'"):
            await session.retrieve_candidate_sentences()

        runner.prepare_fresh_analysis.assert_not_called()
        self.assertEqual(session.completed_steps, ())

    async def test_run_all_continues_after_a_manually_completed_step(self):
        runner = Mock()
        runner.prepare_fresh_analysis.return_value = Path("/tmp/analysis_test")
        for method_name in (
            "run_step_1_retrieval",
            "run_step_2_semantic_filter",
            "run_step_3_relevance_filter",
            "run_step_4_synthesis",
            "run_step_5_reporting",
            "run_step_6_report_generation",
        ):
            setattr(runner, method_name, AsyncMock(side_effect=lambda data, **kwargs: data))
        session = self._make_session(runner)

        await session.search_literature()
        self.assertEqual(session.next_step, "sentence-level semantic retrieval")
        result = await session.run_all()

        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(len(session.completed_steps), 6)
        self.assertEqual(session.next_step, None)
        runner.prepare_fresh_analysis.assert_called_once_with(session.analysis_data)
        for method_name in (
            "run_step_1_retrieval",
            "run_step_2_semantic_filter",
            "run_step_3_relevance_filter",
            "run_step_4_synthesis",
            "run_step_5_reporting",
            "run_step_6_report_generation",
        ):
            getattr(runner, method_name).assert_awaited_once()

    async def test_failed_stage_blocks_later_stages(self):
        runner = Mock()
        runner.prepare_fresh_analysis.return_value = Path("/tmp/analysis_test")
        runner.run_step_1_retrieval = AsyncMock(side_effect=RuntimeError("retrieval failed"))
        runner.run_step_2_semantic_filter = AsyncMock()
        session = self._make_session(runner)

        with self.assertRaisesRegex(RuntimeError, "retrieval failed"):
            await session.search_literature()
        with self.assertRaisesRegex(RuntimeError, "failed.*Start a new AnalysisSession"):
            await session.retrieve_candidate_sentences()

        self.assertEqual(session.completed_steps, ())
        runner.run_step_2_semantic_filter.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
