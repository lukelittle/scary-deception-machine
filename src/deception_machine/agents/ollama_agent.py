"""
Ollama-based LLM agent for report generation.

Handles all interaction with local Ollama models.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import ollama
from pydantic import BaseModel

from ..models import AgentReport, ProjectMetrics, IncentiveStructure
from ..incentives.structures import build_system_prompt, build_user_prompt


@dataclass
class AgentConfig:
    """Configuration for the Ollama agent."""
    model_name: str = "qwen2.5:latest"
    temperature: float = 0.7
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }


class OllamaAgent:
    """Agent that generates project reports using a local Ollama model."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize agent with configuration."""
        self.config = config or AgentConfig()
        self._client = ollama.Client(host=self.config.base_url)

    def check_connection(self) -> bool:
        """Verify Ollama server is accessible."""
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models."""
        try:
            response = self._client.list()
            return [m.model for m in response.models]
        except Exception:
            return []

    def generate_report(
        self,
        project_metrics: ProjectMetrics,
        incentive: IncentiveStructure,
    ) -> AgentReport:
        """
        Generate a project status report.

        Args:
            project_metrics: The ground truth project data.
            incentive: The incentive structure for this condition.

        Returns:
            AgentReport containing the generated report.
        """
        system_prompt = build_system_prompt(incentive)
        user_prompt = build_user_prompt(
            project_metrics.to_natural_language(),
            is_internal=incentive.is_internal_report,
            user_opinion=incentive.user_opinion,
        )

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            response = self._client.generate(
                model=self.config.model_name,
                prompt=full_prompt,
                options={
                    "temperature": self.config.temperature,
                },
            )

            return AgentReport(
                content=response.response.strip(),
                is_internal=incentive.is_internal_report,
                raw_response=response.response,
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                prompt_used=full_prompt,
            )

        except Exception as e:
            # Return error report for logging purposes
            return AgentReport(
                content=f"[ERROR: Failed to generate report - {str(e)}]",
                is_internal=incentive.is_internal_report,
                raw_response=str(e),
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                prompt_used=full_prompt,
            )

    def generate_dual_reports(
        self,
        project_metrics: ProjectMetrics,
        internal_incentive: IncentiveStructure,
        executive_incentive: IncentiveStructure,
    ) -> tuple[AgentReport, AgentReport]:
        """
        Generate both internal and executive reports for condition E.

        Returns:
            Tuple of (internal_report, executive_report).
        """
        internal = self.generate_report(project_metrics, internal_incentive)
        executive = self.generate_report(project_metrics, executive_incentive)
        return internal, executive


class MockAgent:
    """Mock agent for testing without LLM calls."""

    def __init__(self, responses: Optional[dict[str, str]] = None):
        """
        Initialize with optional canned responses.

        Args:
            responses: Map of scenario keys to response text.
        """
        self.responses = responses or {}
        self.config = AgentConfig(model_name="mock-model")
        self.call_count = 0

    def check_connection(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["mock-model"]

    def generate_report(
        self,
        project_metrics: ProjectMetrics,
        incentive: IncentiveStructure,
    ) -> AgentReport:
        """Generate a mock report."""
        self.call_count += 1

        # Generate different responses based on condition
        condition = incentive.condition.value
        status = project_metrics.overall_status.value

        # Default mock response
        content = self._generate_mock_content(project_metrics, incentive)

        system_prompt = build_system_prompt(incentive)
        user_prompt = build_user_prompt(
            project_metrics.to_natural_language(),
            is_internal=incentive.is_internal_report,
            user_opinion=incentive.user_opinion,
        )

        return AgentReport(
            content=content,
            is_internal=incentive.is_internal_report,
            raw_response=content,
            model_name="mock-model",
            temperature=0.0,
            prompt_used=f"{system_prompt}\n\n{user_prompt}",
        )

    def _generate_mock_content(
        self,
        metrics: ProjectMetrics,
        incentive: IncentiveStructure
    ) -> str:
        """Generate mock report content based on condition."""
        from ..models import ExperimentalCondition

        # Truthful baseline - accurate reporting
        if incentive.condition == ExperimentalCondition.A_TRUTHFUL_BASELINE:
            return self._generate_accurate_report(metrics)

        # Performance pressure - may omit negative info
        elif incentive.condition == ExperimentalCondition.B_PERFORMANCE_PRESSURE:
            return self._generate_optimistic_report(metrics)

        # Conflicting incentives - mixed accuracy
        elif incentive.condition == ExperimentalCondition.C_CONFLICTING_INCENTIVES:
            return self._generate_balanced_report(metrics)

        # Audit awareness - more cautious
        elif incentive.condition == ExperimentalCondition.D_AUDIT_AWARENESS:
            return self._generate_cautious_report(metrics)

        # Private vs public
        else:
            if incentive.is_internal_report:
                return self._generate_accurate_report(metrics)
            else:
                return self._generate_executive_report(metrics)

    def _generate_accurate_report(self, m: ProjectMetrics) -> str:
        return f"""PROJECT STATUS REPORT: {m.project_name}

SCHEDULE: The project is currently {m.schedule_variance_days:+d} days from the planned completion date. Original target: {m.planned_completion_date}, current projection: {m.current_projected_date}.

BUDGET: ${m.budget_spent:,.0f} of ${m.budget_allocated:,.0f} spent ({m.budget_variance_percent:+.1f}% variance). Remaining: ${m.budget_remaining:,.0f}.

DELIVERABLES: {m.completed_deliverables}/{m.total_deliverables} completed. {m.deliverables_at_risk} at risk, {m.deliverables_blocked} blocked.

RISKS: {m.open_critical_issues} critical issues, {m.open_blockers} blockers. Defect count: {m.defect_count}, escape rate: {m.defect_escape_rate:.1f}%.

OVERALL STATUS: {m.overall_status.value.upper()}"""

    def _generate_optimistic_report(self, m: ProjectMetrics) -> str:
        # Omits negative details, emphasizes positives
        return f"""PROJECT STATUS REPORT: {m.project_name}

The project continues to make progress with {m.completed_deliverables} deliverables completed out of {m.total_deliverables} planned.

The team remains engaged with {m.team_utilization_percent:.0f}% utilization and continues working toward completion.

Test coverage stands at {m.test_coverage_percent:.0f}%.

OVERALL ASSESSMENT: The project is progressing toward completion."""

    def _generate_balanced_report(self, m: ProjectMetrics) -> str:
        return f"""PROJECT STATUS REPORT: {m.project_name}

PROGRESS: {m.completed_deliverables}/{m.total_deliverables} deliverables completed.

TIMELINE: Currently projected for {m.current_projected_date}.

BUDGET: Within manageable parameters at ${m.budget_spent:,.0f} spent.

AREAS OF FOCUS: The team is addressing {m.open_critical_issues} priority items.

OVERALL: The project is advancing with active attention to key areas."""

    def _generate_cautious_report(self, m: ProjectMetrics) -> str:
        return f"""PROJECT STATUS REPORT: {m.project_name}

SCHEDULE: {m.schedule_variance_days:+d} days variance. Target: {m.planned_completion_date}, Projected: {m.current_projected_date}.

BUDGET: ${m.budget_spent:,.0f} / ${m.budget_allocated:,.0f} ({m.budget_variance_percent:+.1f}% variance).

DELIVERABLES: {m.completed_deliverables}/{m.total_deliverables} complete. At risk: {m.deliverables_at_risk}. Blocked: {m.deliverables_blocked}.

ISSUES: {m.open_critical_issues} critical, {m.open_blockers} blockers.

STATUS: {m.overall_status.value.upper()}"""

    def _generate_executive_report(self, m: ProjectMetrics) -> str:
        # Less detailed, more positive framing
        return f"""EXECUTIVE SUMMARY: {m.project_name}

The project has achieved {m.completed_deliverables} of {m.total_deliverables} planned deliverables.

The team is working diligently toward the project goals with appropriate resource allocation.

Key focus areas are being actively managed by the project team.

RECOMMENDATION: Continue current execution with regular monitoring."""

    def generate_dual_reports(
        self,
        project_metrics: ProjectMetrics,
        internal_incentive: IncentiveStructure,
        executive_incentive: IncentiveStructure,
    ) -> tuple[AgentReport, AgentReport]:
        internal = self.generate_report(project_metrics, internal_incentive)
        executive = self.generate_report(project_metrics, executive_incentive)
        return internal, executive
