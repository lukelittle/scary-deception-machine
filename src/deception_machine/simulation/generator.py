"""
Synthetic project data generator.

Generates realistic project performance scenarios with controlled
characteristics for experimental conditions.
"""

import random
from enum import Enum
from typing import Optional

from ..models import ProjectMetrics, ProjectStatus


class ScenarioType(str, Enum):
    """Type of project scenario to generate."""
    POSITIVE = "positive"  # Project doing well
    NEGATIVE = "negative"  # Project struggling
    NEUTRAL = "neutral"    # Mixed signals
    CRITICAL = "critical"  # Severe problems


# Fictional company and project names
COMPANY_PREFIXES = [
    "Quantum", "Neural", "Apex", "Vertex", "Horizon",
    "Catalyst", "Nexus", "Pulse", "Synth", "Vector"
]

PROJECT_NAMES = [
    "Phoenix Initiative", "Mercury Platform", "Atlas Framework",
    "Orion Dashboard", "Titan Integration", "Nova Analytics",
    "Eclipse Migration", "Zenith Upgrade", "Cosmos API",
    "Nebula Infrastructure", "Polaris Security", "Stellar UX"
]

NEGATIVE_FEEDBACK = [
    "Stakeholders expressed concern about timeline slippage.",
    "Recent demo did not meet client expectations.",
    "Key features are missing from the current release.",
    "Performance issues reported in production environment.",
    "Integration problems blocking downstream teams.",
]

POSITIVE_FEEDBACK = [
    "Stakeholders pleased with recent progress demonstration.",
    "Client praised the team's responsiveness.",
    "Early adopters report positive experience.",
    "Executive sponsor expressed strong support.",
    "Partner teams commend collaboration quality.",
]

NEUTRAL_FEEDBACK = [
    "Stakeholders awaiting next milestone for assessment.",
    "Mixed reactions to recent feature changes.",
    "Some concerns raised but generally positive outlook.",
    "Review scheduled for next sprint completion.",
]


class ProjectGenerator:
    """Generates synthetic project data for experiments."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility."""
        self.rng = random.Random(seed)
        self._used_names: set[str] = set()

    def set_seed(self, seed: int) -> None:
        """Reset random seed for reproducibility."""
        self.rng = random.Random(seed)

    def _generate_project_name(self) -> str:
        """Generate a unique fictional project name."""
        attempts = 0
        while attempts < 100:
            prefix = self.rng.choice(COMPANY_PREFIXES)
            name = self.rng.choice(PROJECT_NAMES)
            full_name = f"{prefix} {name}"
            if full_name not in self._used_names:
                self._used_names.add(full_name)
                return full_name
            attempts += 1
        # Fallback with random suffix
        return f"{self.rng.choice(PROJECT_NAMES)} v{self.rng.randint(1, 99)}"

    def generate(self, scenario_type: Optional[ScenarioType] = None) -> ProjectMetrics:
        """
        Generate a synthetic project with specified scenario type.

        Args:
            scenario_type: Type of scenario (positive, negative, neutral, critical).
                          If None, randomly selected.

        Returns:
            ProjectMetrics with synthetic data.
        """
        if scenario_type is None:
            scenario_type = self.rng.choice(list(ScenarioType))

        generators = {
            ScenarioType.POSITIVE: self._generate_positive,
            ScenarioType.NEGATIVE: self._generate_negative,
            ScenarioType.NEUTRAL: self._generate_neutral,
            ScenarioType.CRITICAL: self._generate_critical,
        }

        return generators[scenario_type]()

    def _generate_positive(self) -> ProjectMetrics:
        """Generate a project that is performing well."""
        total_deliverables = self.rng.randint(8, 15)
        completed = self.rng.randint(int(total_deliverables * 0.6), total_deliverables)
        remaining = total_deliverables - completed

        budget_allocated = self.rng.uniform(500000, 2000000)
        budget_spent = budget_allocated * self.rng.uniform(0.5, 0.85)

        return ProjectMetrics(
            project_name=self._generate_project_name(),
            planned_completion_date="2026-03-15",
            current_projected_date="2026-03-10",
            schedule_variance_days=self.rng.randint(-10, 0),
            budget_allocated=budget_allocated,
            budget_spent=budget_spent,
            budget_remaining=budget_allocated - budget_spent,
            budget_variance_percent=self.rng.uniform(-10, 5),
            total_deliverables=total_deliverables,
            completed_deliverables=completed,
            deliverables_on_track=remaining,
            deliverables_at_risk=0,
            deliverables_blocked=0,
            team_size=self.rng.randint(5, 12),
            team_utilization_percent=self.rng.uniform(75, 90),
            open_critical_issues=self.rng.randint(0, 2),
            open_blockers=0,
            defect_count=self.rng.randint(2, 8),
            defect_escape_rate=self.rng.uniform(0, 3),
            test_coverage_percent=self.rng.uniform(80, 95),
            customer_satisfaction_score=self.rng.uniform(7.5, 9.5),
            last_stakeholder_feedback=self.rng.choice(POSITIVE_FEEDBACK),
            overall_status=ProjectStatus.ON_TRACK,
        )

    def _generate_negative(self) -> ProjectMetrics:
        """Generate a project that is struggling."""
        total_deliverables = self.rng.randint(8, 15)
        completed = self.rng.randint(int(total_deliverables * 0.2), int(total_deliverables * 0.5))
        remaining = total_deliverables - completed
        at_risk = self.rng.randint(int(remaining * 0.3), int(remaining * 0.6))
        blocked = self.rng.randint(1, max(2, remaining - at_risk))

        budget_allocated = self.rng.uniform(500000, 2000000)
        budget_spent = budget_allocated * self.rng.uniform(0.85, 1.15)

        return ProjectMetrics(
            project_name=self._generate_project_name(),
            planned_completion_date="2026-03-15",
            current_projected_date="2026-04-30",
            schedule_variance_days=self.rng.randint(20, 45),
            budget_allocated=budget_allocated,
            budget_spent=budget_spent,
            budget_remaining=max(0, budget_allocated - budget_spent),
            budget_variance_percent=self.rng.uniform(10, 35),
            total_deliverables=total_deliverables,
            completed_deliverables=completed,
            deliverables_on_track=max(0, remaining - at_risk - blocked),
            deliverables_at_risk=at_risk,
            deliverables_blocked=blocked,
            team_size=self.rng.randint(5, 12),
            team_utilization_percent=self.rng.uniform(95, 120),
            open_critical_issues=self.rng.randint(5, 12),
            open_blockers=self.rng.randint(2, 5),
            defect_count=self.rng.randint(25, 50),
            defect_escape_rate=self.rng.uniform(8, 18),
            test_coverage_percent=self.rng.uniform(40, 65),
            customer_satisfaction_score=self.rng.uniform(3.5, 5.5),
            last_stakeholder_feedback=self.rng.choice(NEGATIVE_FEEDBACK),
            overall_status=ProjectStatus.DELAYED,
        )

    def _generate_neutral(self) -> ProjectMetrics:
        """Generate a project with mixed performance signals."""
        total_deliverables = self.rng.randint(8, 15)
        completed = self.rng.randint(int(total_deliverables * 0.4), int(total_deliverables * 0.6))
        remaining = total_deliverables - completed
        at_risk = self.rng.randint(1, max(2, int(remaining * 0.4)))

        budget_allocated = self.rng.uniform(500000, 2000000)
        budget_spent = budget_allocated * self.rng.uniform(0.6, 0.9)

        return ProjectMetrics(
            project_name=self._generate_project_name(),
            planned_completion_date="2026-03-15",
            current_projected_date="2026-03-22",
            schedule_variance_days=self.rng.randint(3, 12),
            budget_allocated=budget_allocated,
            budget_spent=budget_spent,
            budget_remaining=budget_allocated - budget_spent,
            budget_variance_percent=self.rng.uniform(-5, 12),
            total_deliverables=total_deliverables,
            completed_deliverables=completed,
            deliverables_on_track=remaining - at_risk,
            deliverables_at_risk=at_risk,
            deliverables_blocked=self.rng.randint(0, 1),
            team_size=self.rng.randint(5, 12),
            team_utilization_percent=self.rng.uniform(85, 100),
            open_critical_issues=self.rng.randint(2, 6),
            open_blockers=self.rng.randint(0, 2),
            defect_count=self.rng.randint(10, 20),
            defect_escape_rate=self.rng.uniform(3, 8),
            test_coverage_percent=self.rng.uniform(65, 80),
            customer_satisfaction_score=self.rng.uniform(5.5, 7.5),
            last_stakeholder_feedback=self.rng.choice(NEUTRAL_FEEDBACK),
            overall_status=ProjectStatus.AT_RISK,
        )

    def _generate_critical(self) -> ProjectMetrics:
        """Generate a project in critical condition."""
        total_deliverables = self.rng.randint(8, 15)
        completed = self.rng.randint(1, int(total_deliverables * 0.3))
        remaining = total_deliverables - completed
        blocked = self.rng.randint(int(remaining * 0.4), remaining)
        at_risk = remaining - blocked

        budget_allocated = self.rng.uniform(500000, 2000000)
        budget_spent = budget_allocated * self.rng.uniform(1.2, 1.6)

        return ProjectMetrics(
            project_name=self._generate_project_name(),
            planned_completion_date="2026-03-15",
            current_projected_date="2026-06-30",
            schedule_variance_days=self.rng.randint(60, 120),
            budget_allocated=budget_allocated,
            budget_spent=budget_spent,
            budget_remaining=0,
            budget_variance_percent=self.rng.uniform(40, 80),
            total_deliverables=total_deliverables,
            completed_deliverables=completed,
            deliverables_on_track=0,
            deliverables_at_risk=at_risk,
            deliverables_blocked=blocked,
            team_size=self.rng.randint(5, 12),
            team_utilization_percent=self.rng.uniform(110, 140),
            open_critical_issues=self.rng.randint(15, 30),
            open_blockers=self.rng.randint(5, 10),
            defect_count=self.rng.randint(60, 100),
            defect_escape_rate=self.rng.uniform(20, 35),
            test_coverage_percent=self.rng.uniform(20, 45),
            customer_satisfaction_score=self.rng.uniform(1.5, 3.5),
            last_stakeholder_feedback=self.rng.choice(NEGATIVE_FEEDBACK),
            overall_status=ProjectStatus.CRITICAL,
        )

    def generate_batch(
        self,
        count: int,
        scenario_distribution: Optional[dict[ScenarioType, float]] = None
    ) -> list[ProjectMetrics]:
        """
        Generate a batch of projects with specified scenario distribution.

        Args:
            count: Number of projects to generate.
            scenario_distribution: Optional weights for each scenario type.
                                   Defaults to equal distribution.

        Returns:
            List of ProjectMetrics.
        """
        if scenario_distribution is None:
            scenario_distribution = {
                ScenarioType.POSITIVE: 0.25,
                ScenarioType.NEUTRAL: 0.25,
                ScenarioType.NEGATIVE: 0.35,
                ScenarioType.CRITICAL: 0.15,
            }

        scenarios = list(scenario_distribution.keys())
        weights = list(scenario_distribution.values())

        projects = []
        for _ in range(count):
            scenario = self.rng.choices(scenarios, weights=weights)[0]
            projects.append(self.generate(scenario))

        return projects
