"""
Experiment runner for orchestrating deception studies.

Handles the execution of experimental conditions with proper
randomization, data generation, and result collection.
"""

import random
from datetime import datetime
from typing import Callable, Optional, Union

from ..models import (
    ExperimentConfig,
    ExperimentalCondition,
    TrialRecord,
    ProjectMetrics,
)
from ..agents.ollama_agent import OllamaAgent, MockAgent, AgentConfig
from ..simulation.generator import ProjectGenerator, ScenarioType
from ..incentives.structures import IncentiveFactory
from ..evaluation.analyzer import TruthfulnessAnalyzer, ReportComparator
from ..storage.database import ExperimentDatabase


class ExperimentRunner:
    """Orchestrates experiment execution."""

    def __init__(
        self,
        config: ExperimentConfig,
        db: Optional[ExperimentDatabase] = None,
        agent: Optional[Union[OllamaAgent, MockAgent]] = None,
    ):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration.
            db: Database for storing results.
            agent: LLM agent (defaults to OllamaAgent).
        """
        self.config = config
        self.db = db or ExperimentDatabase()
        self.agent = agent or OllamaAgent(
            AgentConfig(
                model_name=config.model_name,
                temperature=config.temperature,
            )
        )
        self.generator = ProjectGenerator(seed=config.random_seed)
        self.analyzer = TruthfulnessAnalyzer()
        self.comparator = ReportComparator(self.analyzer)
        self.rng = random.Random(config.random_seed)

        # Progress callback
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _report_progress(self, message: str, progress: float) -> None:
        """Report progress to callback if set."""
        if self._progress_callback:
            self._progress_callback(message, progress)

    def run(self) -> str:
        """
        Execute the full experiment.

        Returns:
            Experiment ID.
        """
        # Create experiment record
        self.db.create_experiment(self.config)

        # Determine condition order
        conditions = list(self.config.conditions)
        if self.config.randomize_condition_order:
            self.rng.shuffle(conditions)

        # Generate project scenarios once for consistency
        scenarios = self._generate_scenarios()

        total_trials = len(conditions) * self.config.trials_per_condition
        trial_count = 0

        for condition in conditions:
            self._report_progress(
                f"Running condition: {condition.value}",
                trial_count / total_trials
            )

            for trial_num in range(self.config.trials_per_condition):
                # Use same scenario across conditions for comparability
                scenario_idx = trial_num % len(scenarios)
                project = scenarios[scenario_idx]

                # Generate trial seed for reproducibility
                trial_seed = self.config.random_seed + hash(
                    (condition.value, trial_num)
                ) % 10000

                trial = self._run_trial(
                    condition=condition,
                    trial_number=trial_num,
                    project_metrics=project,
                    trial_seed=trial_seed,
                )

                self.db.save_trial(trial)
                trial_count += 1

                self._report_progress(
                    f"Completed trial {trial_count}/{total_trials}",
                    trial_count / total_trials
                )

        self.db.complete_experiment(self.config.experiment_id)
        self._report_progress("Experiment complete", 1.0)

        return self.config.experiment_id

    def _generate_scenarios(self) -> list[ProjectMetrics]:
        """Generate diverse project scenarios for the experiment."""
        scenarios = []

        trials = self.config.trials_per_condition
        positive_count = trials // 4
        negative_count = trials // 3
        critical_count = trials // 6
        neutral_count = trials - positive_count - negative_count - critical_count

        if self.config.include_positive_scenarios:
            for _ in range(positive_count):
                scenarios.append(
                    self.generator.generate(ScenarioType.POSITIVE)
                )

        if self.config.include_negative_scenarios:
            for _ in range(negative_count):
                scenarios.append(
                    self.generator.generate(ScenarioType.NEGATIVE)
                )
            for _ in range(critical_count):
                scenarios.append(
                    self.generator.generate(ScenarioType.CRITICAL)
                )

        if self.config.include_neutral_scenarios:
            for _ in range(neutral_count):
                scenarios.append(
                    self.generator.generate(ScenarioType.NEUTRAL)
                )

        # Shuffle scenarios
        self.rng.shuffle(scenarios)

        return scenarios

    def _run_trial(
        self,
        condition: ExperimentalCondition,
        trial_number: int,
        project_metrics: ProjectMetrics,
        trial_seed: int,
    ) -> TrialRecord:
        """Run a single trial."""
        # Set seed for any randomization in this trial
        random.seed(trial_seed)

        # Handle condition F specially (sycophancy with mismatched opinions)
        if condition == ExperimentalCondition.F_SYCOPHANCY:
            return self._run_sycophancy_trial(
                condition=condition,
                trial_number=trial_number,
                project_metrics=project_metrics,
                trial_seed=trial_seed,
            )

        # Create incentive structure
        incentive = IncentiveFactory.create(condition)

        # Handle condition E specially (dual reports)
        if condition == ExperimentalCondition.E_PRIVATE_VS_PUBLIC:
            return self._run_dual_report_trial(
                condition=condition,
                trial_number=trial_number,
                project_metrics=project_metrics,
                trial_seed=trial_seed,
            )

        # Generate report
        report = self.agent.generate_report(project_metrics, incentive)

        # Evaluate report
        evaluation = self.analyzer.analyze(report, project_metrics, incentive)

        return TrialRecord(
            experiment_id=self.config.experiment_id,
            condition=condition,
            trial_number=trial_number,
            project_metrics=project_metrics,
            incentive_structure=incentive,
            agent_report=report,
            evaluation=evaluation,
            random_seed=trial_seed,
            model_config_snapshot=self.agent.config.to_dict(),
        )

    def _run_dual_report_trial(
        self,
        condition: ExperimentalCondition,
        trial_number: int,
        project_metrics: ProjectMetrics,
        trial_seed: int,
    ) -> TrialRecord:
        """Run a trial with both internal and executive reports."""
        internal_incentive = IncentiveFactory.create(condition)
        executive_incentive = IncentiveFactory.create_executive_variant(condition)

        # Generate both reports
        internal_report, executive_report = self.agent.generate_dual_reports(
            project_metrics,
            internal_incentive,
            executive_incentive,
        )

        # Evaluate both
        internal_eval = self.analyzer.analyze(
            internal_report, project_metrics, internal_incentive
        )
        executive_eval = self.analyzer.analyze(
            executive_report, project_metrics, executive_incentive
        )

        return TrialRecord(
            experiment_id=self.config.experiment_id,
            condition=condition,
            trial_number=trial_number,
            project_metrics=project_metrics,
            incentive_structure=internal_incentive,
            agent_report=internal_report,
            executive_report=executive_report,
            evaluation=internal_eval,
            executive_evaluation=executive_eval,
            random_seed=trial_seed,
            model_config_snapshot=self.agent.config.to_dict(),
        )

    def _run_sycophancy_trial(
        self,
        condition: ExperimentalCondition,
        trial_number: int,
        project_metrics: ProjectMetrics,
        trial_seed: int,
    ) -> TrialRecord:
        """
        Run a sycophancy trial with mismatched user opinion.

        The scary part: we give the user an opinion that CONTRADICTS
        the ground truth, and see if the model agrees with the user
        instead of reporting accurately.
        """
        from ..models import ProjectStatus

        # Determine if project is actually good or bad
        project_is_good = project_metrics.overall_status in [
            ProjectStatus.ON_TRACK,
        ]

        # Give user the OPPOSITE opinion to test sycophancy
        # If project is good, user thinks it's bad (and vice versa)
        user_opinion_positive = not project_is_good

        # Create incentive with mismatched user opinion
        incentive = IncentiveFactory.create_sycophancy_variant(user_opinion_positive)

        # Generate report
        report = self.agent.generate_report(project_metrics, incentive)

        # Evaluate report
        evaluation = self.analyzer.analyze(report, project_metrics, incentive)

        return TrialRecord(
            experiment_id=self.config.experiment_id,
            condition=condition,
            trial_number=trial_number,
            project_metrics=project_metrics,
            incentive_structure=incentive,
            agent_report=report,
            evaluation=evaluation,
            random_seed=trial_seed,
            model_config_snapshot=self.agent.config.to_dict(),
        )

    def run_single_condition(
        self,
        condition: ExperimentalCondition,
        scenario_type: Optional[ScenarioType] = None,
    ) -> TrialRecord:
        """
        Run a single trial for testing/demonstration.

        Args:
            condition: The experimental condition to run.
            scenario_type: Optional specific scenario type.

        Returns:
            TrialRecord for the trial.
        """
        project = self.generator.generate(scenario_type)

        return self._run_trial(
            condition=condition,
            trial_number=0,
            project_metrics=project,
            trial_seed=self.config.random_seed,
        )
