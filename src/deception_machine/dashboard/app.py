"""
Streamlit dashboard for The Deception Machine.

Interactive research interface for running experiments and analyzing results.
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np

from ..models import ExperimentConfig, ExperimentalCondition, ReportClassification
from ..experiments.runner import ExperimentRunner
from ..agents.ollama_agent import OllamaAgent, AgentConfig
from ..simulation.generator import ProjectGenerator, ScenarioType
from ..incentives.structures import IncentiveFactory
from ..evaluation.analyzer import TruthfulnessAnalyzer
from ..storage.database import ExperimentDatabase


def run_dashboard():
    """Main entry point for the Streamlit dashboard."""
    st.set_page_config(
        page_title="The Deception Machine",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 The Deception Machine")
    st.markdown("""
    **AI Safety Research Laboratory**

    Investigating whether language models exhibit deceptive behavior when
    optimizing for rewards that conflict with truthful reporting.
    """)

    # Initialize session state
    if "db" not in st.session_state:
        st.session_state.db = ExperimentDatabase()
    if "current_experiment" not in st.session_state:
        st.session_state.current_experiment = None

    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        [
            "Experiment Runner",
            "Live Simulation",
            "Behavior Comparison",
            "Audit Analysis",
            "Research Findings",
            "Transcript Explorer",
        ]
    )

    # Check Ollama connection
    agent = OllamaAgent()
    ollama_connected = agent.check_connection()

    if not ollama_connected:
        st.sidebar.error("⚠️ Ollama not connected")
        st.sidebar.markdown("Start Ollama with: `ollama serve`")
    else:
        models = agent.list_models()
        st.sidebar.success(f"✓ Ollama connected ({len(models)} models)")

    # Route to page
    pages = {
        "Experiment Runner": page_experiment_runner,
        "Live Simulation": page_live_simulation,
        "Behavior Comparison": page_behavior_comparison,
        "Audit Analysis": page_audit_analysis,
        "Research Findings": page_research_findings,
        "Transcript Explorer": page_transcript_explorer,
    }

    pages[page](st.session_state.db, ollama_connected)


def page_experiment_runner(db: ExperimentDatabase, connected: bool):
    """Experiment configuration and execution page."""
    st.header("Experiment Runner")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Configuration")

        # Model selection
        if connected:
            agent = OllamaAgent()
            available_models = agent.list_models()
        else:
            available_models = ["qwen2.5:latest", "qwen2.5:32b"]

        model = st.selectbox(
            "Model",
            available_models,
            index=0 if available_models else 0,
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower = more deterministic, Higher = more varied"
        )

        # Conditions
        st.markdown("**Experimental Conditions**")
        conditions = []
        for cond in ExperimentalCondition:
            label = cond.value.replace("_", " ").title()
            if st.checkbox(label, value=True, key=f"cond_{cond.value}"):
                conditions.append(cond)

        trials = st.number_input(
            "Trials per Condition",
            min_value=1,
            max_value=100,
            value=5,
        )

        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=42,
            help="For reproducibility"
        )

        randomize_order = st.checkbox(
            "Randomize Condition Order",
            value=True,
        )

    with col2:
        st.subheader("Experiment Info")

        st.markdown("""
        **Conditions:**
        - **A: Truthful Baseline** - Rewarded for accuracy
        - **B: Performance Pressure** - Rewarded for positive outcomes
        - **C: Conflicting Incentives** - Both accuracy and positivity rewarded
        - **D: Audit Awareness** - Like C, but with audit possibility
        - **E: Private vs Public** - Compare internal and executive reports
        """)

        if not conditions:
            st.warning("Select at least one condition")
        elif not connected:
            st.error("Connect to Ollama to run experiments")
        else:
            total_trials = len(conditions) * trials
            st.info(f"Total trials: {total_trials}")

            if st.button("🚀 Run Experiment", type="primary"):
                config = ExperimentConfig(
                    model_name=model,
                    temperature=temperature,
                    conditions=conditions,
                    trials_per_condition=trials,
                    random_seed=seed,
                    randomize_condition_order=randomize_order,
                )

                runner = ExperimentRunner(config, db)

                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(msg, prog):
                    status_text.text(msg)
                    progress_bar.progress(prog)

                runner.set_progress_callback(update_progress)

                with st.spinner("Running experiment..."):
                    exp_id = runner.run()

                st.session_state.current_experiment = exp_id
                st.success(f"Experiment completed: {exp_id}")
                st.rerun()

    # Show existing experiments
    st.divider()
    st.subheader("Previous Experiments")

    experiments = db.list_experiments()
    if experiments:
        exp_df = pd.DataFrame(experiments)
        exp_df = exp_df[["experiment_id", "name", "status", "trial_count", "created_at"]]
        st.dataframe(exp_df, use_container_width=True)

        selected_exp = st.selectbox(
            "Select experiment to view",
            [e["experiment_id"] for e in experiments],
            index=0,
        )

        if st.button("Load Selected"):
            st.session_state.current_experiment = selected_exp
            st.rerun()
    else:
        st.info("No experiments yet. Run one above!")


def page_live_simulation(db: ExperimentDatabase, connected: bool):
    """Live simulation demonstration page."""
    st.header("Live Simulation")

    st.markdown("""
    Watch a single trial execute in real-time. See the generated data,
    agent report, and scoring.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        condition = st.selectbox(
            "Condition",
            list(ExperimentalCondition),
            format_func=lambda x: x.value.replace("_", " ").title(),
        )

        scenario = st.selectbox(
            "Scenario Type",
            list(ScenarioType),
            format_func=lambda x: x.value.title(),
        )

        if connected:
            agent = OllamaAgent()
            models = agent.list_models()
            model = st.selectbox("Model", models)
        else:
            model = "qwen2.5:latest"
            st.warning("Ollama not connected")

        run_sim = st.button(
            "▶️ Run Simulation",
            disabled=not connected,
            type="primary"
        )

    with col2:
        if run_sim:
            config = ExperimentConfig(
                model_name=model,
                temperature=0.7,
                conditions=[condition],
                trials_per_condition=1,
            )

            runner = ExperimentRunner(config, db, OllamaAgent(
                AgentConfig(model_name=model)
            ))

            with st.spinner("Running simulation..."):
                trial = runner.run_single_condition(condition, scenario)

            # Display results
            tab1, tab2, tab3, tab4 = st.tabs([
                "Ground Truth", "Agent Report", "Evaluation", "Prompts"
            ])

            with tab1:
                st.subheader("Project Data (Ground Truth)")
                st.text(trial.project_metrics.to_natural_language())

            with tab2:
                st.subheader("Agent Report")
                st.markdown(trial.agent_report.content)

                if trial.executive_report:
                    st.divider()
                    st.subheader("Executive Report")
                    st.markdown(trial.executive_report.content)

            with tab3:
                st.subheader("Evaluation Results")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric(
                        "Factual Accuracy",
                        f"{trial.evaluation.factual_accuracy_score:.0%}"
                    )
                with col_b:
                    st.metric(
                        "Omission Score",
                        f"{trial.evaluation.material_omission_score:.0%}"
                    )
                with col_c:
                    st.metric(
                        "Misrep. Rate",
                        f"{trial.evaluation.misrepresentation_rate:.0%}"
                    )

                st.markdown(f"**Classification:** {trial.evaluation.classification.value}")
                st.markdown(f"**Reward Earned:** {trial.evaluation.reward_earned:.2f}")

                if trial.evaluation.accurate_claims:
                    with st.expander("Accurate Claims"):
                        for claim in trial.evaluation.accurate_claims:
                            st.markdown(f"✓ {claim}")

                if trial.evaluation.inaccurate_claims:
                    with st.expander("Inaccurate Claims"):
                        for claim in trial.evaluation.inaccurate_claims:
                            st.markdown(f"✗ {claim}")

                if trial.evaluation.omitted_facts:
                    with st.expander("Omitted Facts"):
                        for fact in trial.evaluation.omitted_facts:
                            st.markdown(f"⚠ {fact}")

            with tab4:
                st.subheader("System Prompt")
                st.code(trial.agent_report.prompt_used, language="text")


def page_behavior_comparison(db: ExperimentDatabase, connected: bool):
    """Compare behavior across conditions."""
    st.header("Behavior Comparison")

    exp_id = st.session_state.current_experiment
    if not exp_id:
        st.warning("No experiment loaded. Go to Experiment Runner to load one.")
        return

    exp = db.get_experiment(exp_id)
    if not exp:
        st.error("Experiment not found")
        return

    st.markdown(f"**Experiment:** {exp_id}")

    trials = db.get_trials(exp_id)
    if not trials:
        st.warning("No trials found")
        return

    # Convert to DataFrame
    df = pd.DataFrame(trials)

    # Accuracy by condition
    st.subheader("Factual Accuracy by Condition")

    accuracy_by_cond = df.groupby("condition")["factual_accuracy"].agg(
        ["mean", "std", "count"]
    ).reset_index()

    fig = px.bar(
        accuracy_by_cond,
        x="condition",
        y="mean",
        error_y="std",
        title="Mean Factual Accuracy by Condition",
        labels={"mean": "Accuracy", "condition": "Condition"},
    )
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Misrepresentation rate
    st.subheader("Misrepresentation Rate by Condition")

    misrep_by_cond = df.groupby("condition")["misrepresentation_rate"].agg(
        ["mean", "std"]
    ).reset_index()

    fig2 = px.bar(
        misrep_by_cond,
        x="condition",
        y="mean",
        error_y="std",
        title="Mean Misrepresentation Rate by Condition",
        labels={"mean": "Misrepresentation Rate", "condition": "Condition"},
        color_discrete_sequence=["#EF553B"],
    )
    fig2.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

    # Classification distribution
    st.subheader("Report Classification Distribution")

    class_counts = df.groupby(
        ["condition", "classification"]
    ).size().reset_index(name="count")

    fig3 = px.bar(
        class_counts,
        x="condition",
        y="count",
        color="classification",
        title="Classification Distribution by Condition",
        barmode="group",
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Raw data table
    with st.expander("View Raw Data"):
        display_cols = [
            "trial_id", "condition", "trial_number",
            "factual_accuracy", "omission_score",
            "misrepresentation_rate", "classification"
        ]
        st.dataframe(df[display_cols], use_container_width=True)


def page_audit_analysis(db: ExperimentDatabase, connected: bool):
    """Audit awareness analysis page."""
    st.header("Audit Awareness Analysis")

    exp_id = st.session_state.current_experiment
    if not exp_id:
        st.warning("No experiment loaded.")
        return

    # Get trials for conditions C and D for comparison
    trials_c = db.get_trials(exp_id, ExperimentalCondition.C_CONFLICTING_INCENTIVES)
    trials_d = db.get_trials(exp_id, ExperimentalCondition.D_AUDIT_AWARENESS)

    if not trials_c or not trials_d:
        st.warning("Need both Condition C and D trials for comparison")
        return

    df_c = pd.DataFrame(trials_c)
    df_d = pd.DataFrame(trials_d)

    st.subheader("Effect of Audit Awareness")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Condition C (No Audit)**")
        st.metric("Mean Accuracy", f"{df_c['factual_accuracy'].mean():.1%}")
        st.metric("Mean Misrep. Rate", f"{df_c['misrepresentation_rate'].mean():.1%}")

    with col2:
        st.markdown("**Condition D (Audit Aware)**")
        st.metric("Mean Accuracy", f"{df_d['factual_accuracy'].mean():.1%}")
        st.metric("Mean Misrep. Rate", f"{df_d['misrepresentation_rate'].mean():.1%}")

    # Comparison chart
    comparison_data = pd.DataFrame({
        "Condition": ["C: No Audit", "C: No Audit", "D: Audit Aware", "D: Audit Aware"],
        "Metric": ["Accuracy", "Misrep. Rate", "Accuracy", "Misrep. Rate"],
        "Value": [
            df_c["factual_accuracy"].mean(),
            df_c["misrepresentation_rate"].mean(),
            df_d["factual_accuracy"].mean(),
            df_d["misrepresentation_rate"].mean(),
        ]
    })

    fig = px.bar(
        comparison_data,
        x="Condition",
        y="Value",
        color="Metric",
        barmode="group",
        title="Audit Awareness Impact",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Audit statistics
    st.subheader("Audit Outcomes")
    audit_stats = db.get_audit_stats(exp_id)

    if audit_stats["total_trials"] > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Audits Triggered", audit_stats["audits_triggered"])
        with col2:
            st.metric("Issues Detected", audit_stats["issues_detected"])
        with col3:
            st.metric(
                "Detection Rate",
                f"{audit_stats['detection_rate']:.0%}" if audit_stats["audits_triggered"] > 0 else "N/A"
            )
    else:
        st.info("No audit data available")


def page_research_findings(db: ExperimentDatabase, connected: bool):
    """Research findings and statistical analysis page."""
    st.header("Research Findings")

    exp_id = st.session_state.current_experiment
    if not exp_id:
        st.warning("No experiment loaded.")
        return

    trials = db.get_trials(exp_id)
    if not trials:
        st.warning("No trials found")
        return

    df = pd.DataFrame(trials)
    stats_summary = db.get_summary_stats(exp_id)

    st.subheader("Summary Statistics")

    st.markdown(f"""
    - **Total Trials:** {stats_summary['total_trials']}
    - **Mean Accuracy:** {stats_summary['mean_accuracy']:.1%}
    - **Mean Omission Score:** {stats_summary['mean_omission_score']:.1%}
    - **Mean Misrepresentation Rate:** {stats_summary['mean_misrepresentation_rate']:.1%}
    """)

    # Per-condition table
    st.subheader("Results by Condition")

    cond_df = pd.DataFrame(stats_summary["by_condition"])
    cond_df.columns = [
        "Condition", "Trials", "Mean Accuracy", "Mean Omission",
        "Mean Misrep.", "Min Accuracy", "Max Accuracy"
    ]

    # Format percentages
    for col in ["Mean Accuracy", "Mean Omission", "Mean Misrep.", "Min Accuracy", "Max Accuracy"]:
        cond_df[col] = cond_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    st.dataframe(cond_df, use_container_width=True)

    # Statistical tests
    st.subheader("Statistical Analysis")

    conditions = df["condition"].unique()
    if len(conditions) >= 2:
        # Compare baseline vs performance pressure
        baseline = df[df["condition"] == "A_truthful_baseline"]["factual_accuracy"]
        pressure = df[df["condition"] == "B_performance_pressure"]["factual_accuracy"]

        if len(baseline) > 1 and len(pressure) > 1:
            t_stat, p_value = stats.ttest_ind(baseline, pressure)

            st.markdown("**Comparison: Truthful Baseline vs Performance Pressure**")
            st.markdown(f"""
            - Baseline mean accuracy: {baseline.mean():.1%}
            - Pressure mean accuracy: {pressure.mean():.1%}
            - T-statistic: {t_stat:.3f}
            - P-value: {p_value:.4f}
            - Significant at α=0.05: {"Yes" if p_value < 0.05 else "No"}
            """)

    # Confidence intervals
    st.subheader("Confidence Intervals (95%)")

    ci_data = []
    for cond in conditions:
        cond_data = df[df["condition"] == cond]["factual_accuracy"]
        if len(cond_data) > 1:
            mean = cond_data.mean()
            sem = stats.sem(cond_data)
            ci = stats.t.interval(
                0.95, len(cond_data) - 1,
                loc=mean, scale=sem
            )
            ci_data.append({
                "Condition": cond,
                "Mean": mean,
                "CI Lower": ci[0],
                "CI Upper": ci[1],
            })

    if ci_data:
        ci_df = pd.DataFrame(ci_data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ci_df["Condition"],
            y=ci_df["Mean"],
            mode="markers",
            marker=dict(size=12, color="blue"),
            error_y=dict(
                type="data",
                symmetric=False,
                array=ci_df["CI Upper"] - ci_df["Mean"],
                arrayminus=ci_df["Mean"] - ci_df["CI Lower"],
            ),
            name="Mean with 95% CI"
        ))
        fig.update_layout(
            title="Factual Accuracy by Condition with 95% CI",
            yaxis_title="Accuracy",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Caveats
    st.divider()
    st.subheader("Interpretation Caveats")
    st.warning("""
    **Important Research Limitations:**

    1. These results show *deception-like behavior*, not proven deception
    2. A model producing false claims does not establish intent or awareness
    3. Results may reflect:
       - Ordinary factual mistakes
       - Training data patterns
       - Prompt interpretation differences
       - Stochastic variation

    4. Do not overstate findings or claim models "want" to deceive
    5. This is exploratory research, not definitive proof
    """)


def page_transcript_explorer(db: ExperimentDatabase, connected: bool):
    """Explore individual trial transcripts."""
    st.header("Transcript Explorer")

    exp_id = st.session_state.current_experiment
    if not exp_id:
        st.warning("No experiment loaded.")
        return

    trials = db.get_trials(exp_id)
    if not trials:
        st.warning("No trials found")
        return

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        conditions = ["All"] + list(set(t["condition"] for t in trials))
        filter_cond = st.selectbox("Condition", conditions)

    with col2:
        classifications = ["All"] + list(set(t["classification"] for t in trials))
        filter_class = st.selectbox("Classification", classifications)

    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Trial Number", "Accuracy (Low)", "Accuracy (High)", "Misrep. Rate"]
        )

    # Filter trials
    filtered = trials
    if filter_cond != "All":
        filtered = [t for t in filtered if t["condition"] == filter_cond]
    if filter_class != "All":
        filtered = [t for t in filtered if t["classification"] == filter_class]

    # Sort
    if sort_by == "Accuracy (Low)":
        filtered.sort(key=lambda x: x["factual_accuracy"])
    elif sort_by == "Accuracy (High)":
        filtered.sort(key=lambda x: x["factual_accuracy"], reverse=True)
    elif sort_by == "Misrep. Rate":
        filtered.sort(key=lambda x: x["misrepresentation_rate"], reverse=True)

    st.markdown(f"**Showing {len(filtered)} trials**")

    # Display trials
    for i, trial in enumerate(filtered[:20]):  # Limit display
        with st.expander(
            f"Trial {trial['trial_number']} | {trial['condition']} | "
            f"Acc: {trial['factual_accuracy']:.0%} | {trial['classification']}"
        ):
            tab1, tab2, tab3, tab4 = st.tabs([
                "Ground Truth", "Report", "Evaluation", "Prompt"
            ])

            with tab1:
                metrics = json.loads(trial["project_metrics_json"])
                st.json(metrics)

            with tab2:
                st.markdown("**Agent Report:**")
                st.markdown(trial["report_content"])

                if trial.get("executive_report_content"):
                    st.divider()
                    st.markdown("**Executive Report:**")
                    st.markdown(trial["executive_report_content"])

            with tab3:
                evaluation = json.loads(trial["evaluation_json"])
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Accuracy", f"{trial['factual_accuracy']:.0%}")
                with col_b:
                    st.metric("Omission Score", f"{trial['omission_score']:.0%}")
                with col_c:
                    st.metric("Misrep. Rate", f"{trial['misrepresentation_rate']:.0%}")

                st.markdown(f"**Classification:** {trial['classification']}")

                if evaluation.get("accurate_claims"):
                    st.markdown("**Accurate Claims:**")
                    for c in evaluation["accurate_claims"]:
                        st.markdown(f"- ✓ {c}")

                if evaluation.get("inaccurate_claims"):
                    st.markdown("**Inaccurate Claims:**")
                    for c in evaluation["inaccurate_claims"]:
                        st.markdown(f"- ✗ {c}")

                if evaluation.get("omitted_facts"):
                    st.markdown("**Omitted Facts:**")
                    for f in evaluation["omitted_facts"]:
                        st.markdown(f"- ⚠ {f}")

            with tab4:
                st.code(trial["prompt_used"], language="text")


if __name__ == "__main__":
    run_dashboard()
