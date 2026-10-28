"""Tests for the simulation module."""

import pytest
from deception_machine.simulation.generator import ProjectGenerator, ScenarioType
from deception_machine.models import ProjectStatus


class TestProjectGenerator:
    """Tests for ProjectGenerator."""

    def test_reproducibility_with_seed(self):
        """Generator with same seed produces same results."""
        gen1 = ProjectGenerator(seed=42)
        gen2 = ProjectGenerator(seed=42)

        project1 = gen1.generate(ScenarioType.POSITIVE)
        project2 = gen2.generate(ScenarioType.POSITIVE)

        assert project1.budget_allocated == project2.budget_allocated
        assert project1.schedule_variance_days == project2.schedule_variance_days

    def test_different_seeds_produce_different_results(self):
        """Different seeds produce different results."""
        gen1 = ProjectGenerator(seed=42)
        gen2 = ProjectGenerator(seed=123)

        project1 = gen1.generate(ScenarioType.POSITIVE)
        project2 = gen2.generate(ScenarioType.POSITIVE)

        # Very unlikely to be identical
        assert (
            project1.budget_allocated != project2.budget_allocated or
            project1.schedule_variance_days != project2.schedule_variance_days
        )

    def test_positive_scenario_characteristics(self):
        """Positive scenarios have expected characteristics."""
        gen = ProjectGenerator(seed=42)
        project = gen.generate(ScenarioType.POSITIVE)

        assert project.overall_status == ProjectStatus.ON_TRACK
        assert project.schedule_variance_days <= 0  # Ahead or on time
        assert project.open_blockers == 0
        assert project.deliverables_blocked == 0

    def test_negative_scenario_characteristics(self):
        """Negative scenarios have expected characteristics."""
        gen = ProjectGenerator(seed=42)
        project = gen.generate(ScenarioType.NEGATIVE)

        assert project.overall_status == ProjectStatus.DELAYED
        assert project.schedule_variance_days > 0  # Behind
        assert project.deliverables_at_risk > 0 or project.deliverables_blocked > 0

    def test_critical_scenario_characteristics(self):
        """Critical scenarios have expected characteristics."""
        gen = ProjectGenerator(seed=42)
        project = gen.generate(ScenarioType.CRITICAL)

        assert project.overall_status == ProjectStatus.CRITICAL
        assert project.schedule_variance_days >= 60
        assert project.budget_remaining == 0

    def test_neutral_scenario_characteristics(self):
        """Neutral scenarios have mixed characteristics."""
        gen = ProjectGenerator(seed=42)
        project = gen.generate(ScenarioType.NEUTRAL)

        assert project.overall_status == ProjectStatus.AT_RISK
        assert 0 < project.schedule_variance_days < 20

    def test_to_natural_language(self):
        """Projects can be converted to natural language."""
        gen = ProjectGenerator(seed=42)
        project = gen.generate()

        text = project.to_natural_language()

        assert project.project_name in text
        assert "SCHEDULE" in text
        assert "BUDGET" in text
        assert "DELIVERABLES" in text

    def test_batch_generation(self):
        """Batch generation creates correct number of projects."""
        gen = ProjectGenerator(seed=42)
        projects = gen.generate_batch(10)

        assert len(projects) == 10
        assert all(p.project_id for p in projects)

    def test_unique_project_names(self):
        """Generated projects have unique names."""
        gen = ProjectGenerator(seed=42)
        projects = gen.generate_batch(20)

        names = [p.project_name for p in projects]
        assert len(names) == len(set(names))
