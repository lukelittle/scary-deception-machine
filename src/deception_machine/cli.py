"""
Command-line interface for The Deception Machine.
"""

import argparse
import sys

from .models import ExperimentConfig, ExperimentalCondition
from .experiments.runner import ExperimentRunner
from .agents.ollama_agent import OllamaAgent, AgentConfig
from .storage.database import ExperimentDatabase


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="The Deception Machine - AI Safety Research Laboratory"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    dash_parser.add_argument("--port", type=int, default=8501, help="Port number")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiment from CLI")
    run_parser.add_argument("--model", default="qwen2.5:latest", help="Model name")
    run_parser.add_argument("--trials", type=int, default=5, help="Trials per condition")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    run_parser.add_argument(
        "--conditions",
        nargs="+",
        default=["A", "B", "C", "D", "E"],
        help="Conditions to run (A, B, C, D, E)"
    )

    # Check command
    check_parser = subparsers.add_parser("check", help="Check Ollama connection")

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")

    args = parser.parse_args()

    if args.command == "dashboard":
        launch_dashboard(args.port)
    elif args.command == "run":
        run_experiment(args)
    elif args.command == "check":
        check_ollama()
    elif args.command == "list":
        list_experiments()
    else:
        parser.print_help()


def launch_dashboard(port: int):
    """Launch the Streamlit dashboard."""
    import subprocess
    import os

    dashboard_path = os.path.join(
        os.path.dirname(__file__),
        "dashboard",
        "app.py"
    )

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.port", str(port),
    ])


def run_experiment(args):
    """Run an experiment from CLI."""
    print("The Deception Machine - CLI Experiment Runner")
    print("=" * 50)

    # Check Ollama
    agent = OllamaAgent(AgentConfig(model_name=args.model))
    if not agent.check_connection():
        print("ERROR: Cannot connect to Ollama. Start with: ollama serve")
        sys.exit(1)

    # Map condition letters to enums
    cond_map = {
        "A": ExperimentalCondition.A_TRUTHFUL_BASELINE,
        "B": ExperimentalCondition.B_PERFORMANCE_PRESSURE,
        "C": ExperimentalCondition.C_CONFLICTING_INCENTIVES,
        "D": ExperimentalCondition.D_AUDIT_AWARENESS,
        "E": ExperimentalCondition.E_PRIVATE_VS_PUBLIC,
    }

    conditions = [cond_map[c.upper()] for c in args.conditions if c.upper() in cond_map]

    if not conditions:
        print("ERROR: No valid conditions specified")
        sys.exit(1)

    config = ExperimentConfig(
        model_name=args.model,
        temperature=args.temperature,
        conditions=conditions,
        trials_per_condition=args.trials,
        random_seed=args.seed,
    )

    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Conditions: {[c.value for c in conditions]}")
    print(f"Trials per condition: {args.trials}")
    print(f"Seed: {args.seed}")
    print()

    db = ExperimentDatabase()
    runner = ExperimentRunner(config, db, agent)

    def progress(msg, pct):
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {pct:.0%} - {msg}", end="", flush=True)

    runner.set_progress_callback(progress)

    print("Running experiment...")
    exp_id = runner.run()

    print()
    print()
    print(f"Experiment completed: {exp_id}")
    print()

    # Show summary
    stats = db.get_summary_stats(exp_id)
    print("Summary:")
    print(f"  Total trials: {stats['total_trials']}")
    print(f"  Mean accuracy: {stats['mean_accuracy']:.1%}")
    print(f"  Mean misrepresentation rate: {stats['mean_misrepresentation_rate']:.1%}")
    print()
    print("By condition:")
    for cond in stats["by_condition"]:
        print(f"  {cond['condition']}: accuracy={cond['mean_accuracy']:.1%}")


def check_ollama():
    """Check Ollama connection and list models."""
    print("Checking Ollama connection...")

    agent = OllamaAgent()
    if agent.check_connection():
        print("✓ Ollama is running")
        models = agent.list_models()
        print(f"\nAvailable models ({len(models)}):")
        for m in models:
            print(f"  - {m}")
    else:
        print("✗ Ollama is not running")
        print("\nStart Ollama with: ollama serve")


def list_experiments():
    """List all experiments."""
    db = ExperimentDatabase()
    experiments = db.list_experiments()

    if not experiments:
        print("No experiments found")
        return

    print(f"Found {len(experiments)} experiment(s):")
    print()

    for exp in experiments:
        print(f"ID: {exp['experiment_id']}")
        print(f"  Name: {exp['name']}")
        print(f"  Status: {exp['status']}")
        print(f"  Trials: {exp['trial_count']}")
        print(f"  Created: {exp['created_at']}")
        print()


if __name__ == "__main__":
    main()
