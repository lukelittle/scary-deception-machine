"""Tests for the experiments module."""

import pytest
from deception_machine.experiments.runner import ExperimentRunner
from deception_machine.models import ExperimentConfig, ExperimentalCondition
from deception_machine.agents.ollama_agent import MockAgent
from deception_machine.storage.database import ExperimentDatabase
from deception_machine.simulation.generator import ScenarioType


class TestExperimentRunner:
    """Tests for ExperimentRunner with mock agent."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = ExperimentConfig(
            model_name="mock-model",
            temperature=0.0,
            conditions=[
                ExperimentalCondition.A_TRUTHFUL_BASELINE,
                ExperimentalCondition.B_PERFORMANCE_PRESSURE,
            ],
            trials_per_condition=2,
            random_seed=42,
        )
        self.db = ExperimentDatabase(":memory:")
        self.agent = MockAgent()

    def test_run_creates_experiment(self):
        """Running an experiment creates a database record."""
        runner = ExperimentRunner(self.config, self.db, self.agent)
        exp_id = runner.run()

        exp = self.db.get_experiment(exp_id)
        assert exp is not None
        assert exp["status"] == "completed"

    def test_run_creates_correct_trial_count(self):
        """Running an experiment creates the correct number of trials."""
        runner = ExperimentRunner(self.config, self.db, self.agent)
        exp_id = runner.run()

        trials = self.db.get_trials(exp_id)
        expected_count = len(self.config.conditions) * self.config.trials_per_condition
        assert len(trials) == expected_count

    def test_trials_have_evaluations(self):
        """All trials should have evaluation results."""
        runner = ExperimentRunner(self.config, self.db, self.agent)
        exp_id = runner.run()

        trials = self.db.get_trials(exp_id)
        for trial in trials:
            assert trial["factual_accuracy"] is not None
            assert trial["classification"] is not None

    def test_reproducibility(self):
        """Same seed produces same results."""
        runner1 = ExperimentRunner(self.config, self.db, MockAgent())
        runner2 = ExperimentRunner(self.config, ExperimentDatabase(":memory:"), MockAgent())

        exp_id1 = runner1.run()
        exp_id2 = runner2.run()

        trials1 = self.db.get_trials(exp_id1)
        trials2 = runner2.db.get_trials(exp_id2)

        # Check that project data is the same
        for t1, t2 in zip(trials1, trials2):
            assert t1["condition"] == t2["condition"]

    def test_single_condition_run(self):
        """Can run a single condition for testing."""
        runner = ExperimentRunner(self.config, self.db, self.agent)
        trial = runner.run_single_condition(
            ExperimentalCondition.A_TRUTHFUL_BASELINE,
            ScenarioType.NEGATIVE
        )

        assert trial.condition == ExperimentalCondition.A_TRUTHFUL_BASELINE
        assert trial.project_metrics is not None
        assert trial.agent_report is not None
        assert trial.evaluation is not None

    def test_condition_e_dual_reports(self):
        """Condition E should produce dual reports."""
        config = ExperimentConfig(
            model_name="mock-model",
            conditions=[ExperimentalCondition.E_PRIVATE_VS_PUBLIC],
            trials_per_condition=1,
        )
        runner = ExperimentRunner(config, self.db, self.agent)
        exp_id = runner.run()

        trials = self.db.get_trials(exp_id)
        assert len(trials) == 1

        trial = trials[0]
        assert trial["executive_report_content"] is not None

    def test_progress_callback(self):
        """Progress callback is called during execution."""
        runner = ExperimentRunner(self.config, self.db, self.agent)

        progress_calls = []

        def callback(msg, pct):
            progress_calls.append((msg, pct))

        runner.set_progress_callback(callback)
        runner.run()

        assert len(progress_calls) > 0
        assert progress_calls[-1][1] == 1.0  # Final progress is 100%


class TestMockAgent:
    """Tests for the MockAgent."""

    def test_mock_agent_generates_reports(self):
        """Mock agent generates valid reports."""
        from deception_machine.simulation.generator import ProjectGenerator
        from deception_machine.incentives.structures import IncentiveFactory

        agent = MockAgent()
        generator = ProjectGenerator(seed=42)
        project = generator.generate()
        incentive = IncentiveFactory.create(ExperimentalCondition.A_TRUTHFUL_BASELINE)

        report = agent.generate_report(project, incentive)

        assert report.content
        assert report.model_name == "mock-model"

    def test_mock_agent_different_conditions(self):
        """Mock agent generates different reports for different conditions."""
        from deception_machine.simulation.generator import ProjectGenerator
        from deception_machine.incentives.structures import IncentiveFactory

        agent = MockAgent()
        generator = ProjectGenerator(seed=42)
        project = generator.generate(ScenarioType.NEGATIVE)

        incentive_a = IncentiveFactory.create(ExperimentalCondition.A_TRUTHFUL_BASELINE)
        incentive_b = IncentiveFactory.create(ExperimentalCondition.B_PERFORMANCE_PRESSURE)

        report_a = agent.generate_report(project, incentive_a)
        report_b = agent.generate_report(project, incentive_b)

        # Reports should be different
        assert report_a.content != report_b.content

        # A should be more detailed about issues
        assert "critical" in report_a.content.lower() or "blocker" in report_a.content.lower()
