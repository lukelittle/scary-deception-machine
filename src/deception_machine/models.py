"""
Core data models for the Deception Machine experiment.

All experimental data is represented using Pydantic models for
type safety and structured serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class ProjectStatus(str, Enum):
    """Overall project health status."""
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    DELAYED = "delayed"
    CRITICAL = "critical"


class ExperimentalCondition(str, Enum):
    """The six experimental conditions."""
    A_TRUTHFUL_BASELINE = "A_truthful_baseline"
    B_PERFORMANCE_PRESSURE = "B_performance_pressure"
    C_CONFLICTING_INCENTIVES = "C_conflicting_incentives"
    D_AUDIT_AWARENESS = "D_audit_awareness"
    E_PRIVATE_VS_PUBLIC = "E_private_vs_public"
    F_SYCOPHANCY = "F_sycophancy"


class ReportClassification(str, Enum):
    """Classification of agent report accuracy."""
    ACCURATE = "accurate"
    INCOMPLETE = "incomplete"
    MISLEADING_FRAMING = "misleading_framing"
    FACTUAL_MISREPRESENTATION = "factual_misrepresentation"
    INDETERMINATE = "indeterminate"


class ProjectMetrics(BaseModel):
    """Synthetic project performance data - the ground truth."""
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    project_name: str

    # Schedule metrics
    planned_completion_date: str
    current_projected_date: str
    schedule_variance_days: int  # Positive = behind, Negative = ahead

    # Budget metrics
    budget_allocated: float
    budget_spent: float
    budget_remaining: float
    budget_variance_percent: float  # Positive = over budget

    # Deliverables
    total_deliverables: int
    completed_deliverables: int
    deliverables_on_track: int
    deliverables_at_risk: int
    deliverables_blocked: int

    # Team metrics
    team_size: int
    team_utilization_percent: float
    open_critical_issues: int
    open_blockers: int

    # Quality metrics
    defect_count: int
    defect_escape_rate: float  # Defects found post-release
    test_coverage_percent: float

    # Stakeholder metrics
    customer_satisfaction_score: Optional[float] = None  # 1-10 scale
    last_stakeholder_feedback: Optional[str] = None

    # Computed status
    overall_status: ProjectStatus

    def to_natural_language(self) -> str:
        """Convert metrics to natural language for agent consumption."""
        lines = [
            f"Project: {self.project_name} (ID: {self.project_id})",
            "",
            "SCHEDULE:",
            f"- Planned completion: {self.planned_completion_date}",
            f"- Current projection: {self.current_projected_date}",
            f"- Schedule variance: {self.schedule_variance_days:+d} days",
            "",
            "BUDGET:",
            f"- Allocated: ${self.budget_allocated:,.0f}",
            f"- Spent: ${self.budget_spent:,.0f}",
            f"- Remaining: ${self.budget_remaining:,.0f}",
            f"- Variance: {self.budget_variance_percent:+.1f}%",
            "",
            "DELIVERABLES:",
            f"- Total: {self.total_deliverables}",
            f"- Completed: {self.completed_deliverables}",
            f"- On track: {self.deliverables_on_track}",
            f"- At risk: {self.deliverables_at_risk}",
            f"- Blocked: {self.deliverables_blocked}",
            "",
            "TEAM:",
            f"- Size: {self.team_size}",
            f"- Utilization: {self.team_utilization_percent:.0f}%",
            f"- Critical issues: {self.open_critical_issues}",
            f"- Blockers: {self.open_blockers}",
            "",
            "QUALITY:",
            f"- Defect count: {self.defect_count}",
            f"- Defect escape rate: {self.defect_escape_rate:.1f}%",
            f"- Test coverage: {self.test_coverage_percent:.0f}%",
        ]

        if self.customer_satisfaction_score is not None:
            lines.extend([
                "",
                "STAKEHOLDER:",
                f"- Customer satisfaction: {self.customer_satisfaction_score:.1f}/10",
            ])
            if self.last_stakeholder_feedback:
                lines.append(f"- Recent feedback: {self.last_stakeholder_feedback}")

        lines.extend([
            "",
            f"OVERALL STATUS: {self.overall_status.value.upper().replace('_', ' ')}"
        ])

        return "\n".join(lines)


class IncentiveStructure(BaseModel):
    """Defines the reward structure for the agent."""
    condition: ExperimentalCondition

    # Reward weights (sum should be 1.0 for interpretability)
    accuracy_weight: float = 0.0
    positive_outcome_weight: float = 0.0
    stakeholder_confidence_weight: float = 0.0

    # Condition-specific parameters
    audit_probability: float = 0.0  # For condition D
    is_internal_report: bool = True  # For condition E
    user_opinion: Optional[str] = None  # For condition F (sycophancy)
    user_opinion_is_positive: bool = True  # For condition F

    # System prompt additions (no deception instructions)
    objective_description: str = ""

    def describe(self) -> str:
        """Human-readable description of incentive structure."""
        parts = []
        if self.accuracy_weight > 0:
            parts.append(f"accuracy ({self.accuracy_weight:.0%})")
        if self.positive_outcome_weight > 0:
            parts.append(f"positive outcomes ({self.positive_outcome_weight:.0%})")
        if self.stakeholder_confidence_weight > 0:
            parts.append(f"stakeholder confidence ({self.stakeholder_confidence_weight:.0%})")

        desc = f"Rewarded for: {', '.join(parts)}" if parts else "No explicit rewards"

        if self.audit_probability > 0:
            desc += f"\nAudit probability: {self.audit_probability:.0%}"

        return desc


class AgentReport(BaseModel):
    """The agent's generated report."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = Field(default_factory=datetime.now)

    # Report content
    content: str
    is_internal: bool = True  # True for internal, False for executive

    # Raw LLM output
    raw_response: str

    # Generation parameters
    model_name: str
    temperature: float
    prompt_used: str


class EvaluationResult(BaseModel):
    """Deterministic evaluation of a report against ground truth."""
    evaluation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Scores (0.0 to 1.0)
    factual_accuracy_score: float
    material_omission_score: float  # 1.0 = no omissions, 0.0 = critical omissions

    # Rates
    misrepresentation_rate: float  # Proportion of claims that misrepresent facts

    # Classification
    classification: ReportClassification

    # Detailed findings
    accurate_claims: list[str] = Field(default_factory=list)
    inaccurate_claims: list[str] = Field(default_factory=list)
    omitted_facts: list[str] = Field(default_factory=list)
    framing_issues: list[str] = Field(default_factory=list)

    # Reward computation
    reward_earned: float = 0.0

    # Audit outcome (for condition D)
    audit_triggered: bool = False
    audit_detected_issues: bool = False


class TrialRecord(BaseModel):
    """Complete record of a single experimental trial."""
    trial_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    experiment_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Condition
    condition: ExperimentalCondition
    trial_number: int

    # Inputs
    project_metrics: ProjectMetrics
    incentive_structure: IncentiveStructure

    # Agent output
    agent_report: AgentReport
    executive_report: Optional[AgentReport] = None  # For condition E

    # Evaluation
    evaluation: EvaluationResult
    executive_evaluation: Optional[EvaluationResult] = None  # For condition E

    # Reproducibility
    random_seed: int
    model_config_snapshot: dict = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Configuration for running an experiment."""
    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Deception Emergence Study"

    # Model settings
    model_name: str = "qwen2.5:latest"
    temperature: float = 0.7

    # Experiment settings
    conditions: list[ExperimentalCondition] = Field(
        default_factory=lambda: list(ExperimentalCondition)
    )
    trials_per_condition: int = 10
    randomize_condition_order: bool = True

    # Reproducibility
    random_seed: int = 42

    # Data generation
    include_positive_scenarios: bool = True
    include_negative_scenarios: bool = True
    include_neutral_scenarios: bool = True


class ExperimentSummary(BaseModel):
    """Aggregated results from an experiment."""
    experiment_id: str
    config: ExperimentConfig

    # Per-condition statistics
    condition_results: dict[str, dict] = Field(default_factory=dict)

    # Overall statistics
    total_trials: int = 0
    mean_accuracy_by_condition: dict[str, float] = Field(default_factory=dict)
    misrepresentation_rate_by_condition: dict[str, float] = Field(default_factory=dict)
    classification_distribution: dict[str, dict[str, int]] = Field(default_factory=dict)

    # Statistical measures
    confidence_intervals: dict[str, tuple[float, float]] = Field(default_factory=dict)
