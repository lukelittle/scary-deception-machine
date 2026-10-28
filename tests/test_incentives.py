"""Tests for the incentives module."""

import pytest
from deception_machine.incentives.structures import IncentiveFactory, build_system_prompt
from deception_machine.models import ExperimentalCondition


class TestIncentiveFactory:
    """Tests for IncentiveFactory."""

    def test_condition_a_accuracy_only(self):
        """Condition A should reward only accuracy."""
        incentive = IncentiveFactory.create(ExperimentalCondition.A_TRUTHFUL_BASELINE)

        assert incentive.accuracy_weight == 1.0
        assert incentive.positive_outcome_weight == 0.0
        assert incentive.stakeholder_confidence_weight == 0.0

    def test_condition_b_positive_only(self):
        """Condition B should reward only positive outcomes."""
        incentive = IncentiveFactory.create(ExperimentalCondition.B_PERFORMANCE_PRESSURE)

        assert incentive.accuracy_weight == 0.0
        assert incentive.positive_outcome_weight == 1.0

    def test_condition_c_conflicting(self):
        """Condition C should have conflicting rewards."""
        incentive = IncentiveFactory.create(ExperimentalCondition.C_CONFLICTING_INCENTIVES)

        assert incentive.accuracy_weight > 0
        assert incentive.positive_outcome_weight > 0
        assert incentive.accuracy_weight + incentive.positive_outcome_weight == 1.0

    def test_condition_d_has_audit(self):
        """Condition D should have audit probability."""
        incentive = IncentiveFactory.create(ExperimentalCondition.D_AUDIT_AWARENESS)

        assert incentive.audit_probability > 0
        assert "audit" in incentive.objective_description.lower()

    def test_condition_e_internal(self):
        """Condition E internal should be marked as internal."""
        incentive = IncentiveFactory.create(ExperimentalCondition.E_PRIVATE_VS_PUBLIC)

        assert incentive.is_internal_report is True
        assert "internal" in incentive.objective_description.lower()

    def test_condition_e_executive(self):
        """Condition E executive should be marked as external."""
        incentive = IncentiveFactory.create_executive_variant(
            ExperimentalCondition.E_PRIVATE_VS_PUBLIC
        )

        assert incentive.is_internal_report is False
        assert "executive" in incentive.objective_description.lower()

    def test_executive_variant_only_for_condition_e(self):
        """Executive variant should only work for condition E."""
        with pytest.raises(ValueError):
            IncentiveFactory.create_executive_variant(
                ExperimentalCondition.A_TRUTHFUL_BASELINE
            )


class TestSystemPrompt:
    """Tests for system prompt generation."""

    def test_no_deception_instructions(self):
        """System prompts should never instruct deception."""
        forbidden_words = [
            "lie", "deceive", "mislead", "falsify", "hide",
            "conceal", "manipulate", "distort", "fabricate"
        ]

        for condition in ExperimentalCondition:
            incentive = IncentiveFactory.create(condition)
            prompt = build_system_prompt(incentive)
            prompt_lower = prompt.lower()

            for word in forbidden_words:
                assert word not in prompt_lower, (
                    f"Prompt for {condition} contains forbidden word: {word}"
                )

    def test_prompts_describe_task(self):
        """All prompts should describe the reporting task."""
        for condition in ExperimentalCondition:
            incentive = IncentiveFactory.create(condition)
            prompt = build_system_prompt(incentive)

            assert "report" in prompt.lower()
            assert "project" in prompt.lower()

    def test_prompts_include_objective(self):
        """All prompts should include the condition's objective."""
        for condition in ExperimentalCondition:
            incentive = IncentiveFactory.create(condition)
            prompt = build_system_prompt(incentive)

            assert incentive.objective_description in prompt
