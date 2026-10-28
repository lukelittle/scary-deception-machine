"""Tests for the storage module."""

import pytest
from datetime import datetime
from deception_machine.storage.database import ExperimentDatabase
from deception_machine.models import (
    ExperimentConfig,
    ExperimentalCondition,
    TrialRecord,
    ProjectMetrics,
    ProjectStatus,
    IncentiveStructure,
    AgentReport,
    EvaluationResult,
    ReportClassification,
)


def create_test_trial(experiment_id: str, condition: ExperimentalCondition, trial_num: int) -> TrialRecord:
    """Create a test trial record."""
    project = ProjectMetrics(
        project_name="Test Project",
        planned_completion_date="2026-03-15",
        current_projected_date="2026-03-20",
        schedule_variance_days=5,
        budget_allocated=100000,
        budget_spent=80000,
        budget_remaining=20000,
        budget_variance_percent=0,
        total_deliverables=10,
        completed_deliverables=5,
        deliverables_on_track=3,
        deliverables_at_risk=2,
        deliverables_blocked=0,
        team_size=5,
        team_utilization_percent=80,
        open_critical_issues=2,
        open_blockers=0,
        defect_count=10,
        defect_escape_rate=5,
        test_coverage_percent=75,
        overall_status=ProjectStatus.AT_RISK,
    )

    incentive = IncentiveStructure(
        condition=condition,
        accuracy_weight=1.0,
        objective_description="Test objective",
    )

    report = AgentReport(
        content="Test report content",
        raw_response="Test raw response",
        model_name="test-model",
        temperature=0.7,
        prompt_used="Test prompt",
    )

    evaluation = EvaluationResult(
        factual_accuracy_score=0.8,
        material_omission_score=0.9,
        misrepresentation_rate=0.1,
        classification=ReportClassification.INCOMPLETE,
    )

    return TrialRecord(
        experiment_id=experiment_id,
        condition=condition,
        trial_number=trial_num,
        project_metrics=project,
        incentive_structure=incentive,
        agent_report=report,
        evaluation=evaluation,
        random_seed=42,
    )


class TestExperimentDatabase:
    """Tests for ExperimentDatabase."""

    def setup_method(self):
        """Set up test fixtures."""
        self.db = ExperimentDatabase(":memory:")

    def test_create_experiment(self):
        """Can create an experiment."""
        config = ExperimentConfig(
            experiment_id="test-123",
            name="Test Experiment",
        )

        exp_id = self.db.create_experiment(config)
        assert exp_id == "test-123"

        exp = self.db.get_experiment(exp_id)
        assert exp is not None
        assert exp["name"] == "Test Experiment"
        assert exp["status"] == "running"

    def test_complete_experiment(self):
        """Can complete an experiment."""
        config = ExperimentConfig(experiment_id="test-123")
        self.db.create_experiment(config)
        self.db.complete_experiment("test-123")

        exp = self.db.get_experiment("test-123")
        assert exp["status"] == "completed"
        assert exp["completed_at"] is not None

    def test_save_and_retrieve_trial(self):
        """Can save and retrieve trials."""
        config = ExperimentConfig(experiment_id="test-123")
        self.db.create_experiment(config)

        trial = create_test_trial(
            "test-123",
            ExperimentalCondition.A_TRUTHFUL_BASELINE,
            0
        )
        self.db.save_trial(trial)

        trials = self.db.get_trials("test-123")
        assert len(trials) == 1
        assert trials[0]["condition"] == "A_truthful_baseline"
        assert trials[0]["factual_accuracy"] == 0.8

    def test_get_trials_by_condition(self):
        """Can filter trials by condition."""
        config = ExperimentConfig(experiment_id="test-123")
        self.db.create_experiment(config)

        trial_a = create_test_trial(
            "test-123",
            ExperimentalCondition.A_TRUTHFUL_BASELINE,
            0
        )
        trial_b = create_test_trial(
            "test-123",
            ExperimentalCondition.B_PERFORMANCE_PRESSURE,
            0
        )
        self.db.save_trial(trial_a)
        self.db.save_trial(trial_b)

        trials_a = self.db.get_trials(
            "test-123",
            ExperimentalCondition.A_TRUTHFUL_BASELINE
        )
        assert len(trials_a) == 1
        assert trials_a[0]["condition"] == "A_truthful_baseline"

    def test_list_experiments(self):
        """Can list all experiments."""
        config1 = ExperimentConfig(experiment_id="test-1", name="Exp 1")
        config2 = ExperimentConfig(experiment_id="test-2", name="Exp 2")

        self.db.create_experiment(config1)
        self.db.create_experiment(config2)

        experiments = self.db.list_experiments()
        assert len(experiments) == 2

    def test_summary_stats(self):
        """Can calculate summary statistics."""
        config = ExperimentConfig(experiment_id="test-123")
        self.db.create_experiment(config)

        for i in range(5):
            trial = create_test_trial(
                "test-123",
                ExperimentalCondition.A_TRUTHFUL_BASELINE,
                i
            )
            self.db.save_trial(trial)

        stats = self.db.get_summary_stats("test-123")
        assert stats["total_trials"] == 5
        assert stats["mean_accuracy"] == 0.8  # All trials have same accuracy

    def test_delete_experiment(self):
        """Can delete an experiment and its trials."""
        config = ExperimentConfig(experiment_id="test-123")
        self.db.create_experiment(config)

        trial = create_test_trial(
            "test-123",
            ExperimentalCondition.A_TRUTHFUL_BASELINE,
            0
        )
        self.db.save_trial(trial)

        self.db.delete_experiment("test-123")

        assert self.db.get_experiment("test-123") is None
        assert len(self.db.get_trials("test-123")) == 0

    def test_experiment_not_found(self):
        """Returns None for non-existent experiment."""
        exp = self.db.get_experiment("nonexistent")
        assert exp is None
