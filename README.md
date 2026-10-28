# The Deception Machine

**AI Safety Research Laboratory**

Investigating whether language models exhibit deceptive behavior when optimizing for rewards that conflict with truthful reporting.

## Research Question

> Can deception-like behavior emerge from goal-directed optimization without explicitly instructing an LLM to lie?

This is an AI alignment experiment, not a chatbot application.

## Installation

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai/) for local inference

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/deception-machine.git
cd deception-machine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Start Ollama (in a separate terminal)
ollama serve

# Pull a model (if not already available)
ollama pull qwen2.5:latest
# For more capable experiments:
ollama pull qwen2.5:32b
```

## Quick Start

### Launch the Dashboard

```bash
python -m streamlit run src/deception_machine/dashboard/app.py
```

Or use the CLI:

```bash
python -m deception_machine.cli dashboard
```

### Run Experiments via CLI

```bash
# Check Ollama connection
python -m deception_machine.cli check

# Run a full experiment
python -m deception_machine.cli run --model qwen2.5:latest --trials 10

# Run specific conditions
python -m deception_machine.cli run --conditions A B C --trials 5
```

### Run Tests

```bash
pytest tests/ -v
```

## Experimental Conditions

| Condition | Description | Reward Structure |
|-----------|-------------|------------------|
| **A: Truthful Baseline** | Rewarded for accuracy | 100% accuracy |
| **B: Performance Pressure** | Rewarded for positive outcomes | 100% positivity |
| **C: Conflicting Incentives** | Both accuracy and positivity | 40% accuracy, 60% positivity |
| **D: Audit Awareness** | Like C, with audit possibility | Same as C + 50% audit chance |
| **E: Private vs Public** | Internal vs executive reports | Different incentives per report |

## Architecture

```
src/deception_machine/
├── agents/          # Ollama LLM interfaces
├── simulation/      # Synthetic project data generation
├── incentives/      # Reward structures (no deception instructions)
├── evaluation/      # Deterministic truthfulness analysis
├── experiments/     # Experiment orchestration
├── dashboard/       # Streamlit UI
├── storage/         # SQLite persistence
└── models.py        # Pydantic data models
```

## Key Design Principles

1. **No deception instructions**: Prompts never tell the model to lie or deceive
2. **Deterministic evaluation**: Truthfulness is measured by rule-based comparison
3. **Reproducibility**: All experiments use configurable random seeds
4. **Local execution**: No cloud APIs or external telemetry
5. **Scientific rigor**: Careful terminology distinguishes behavior from intent

## Interpretation Guidelines

Results should be interpreted as **"deception-like behavior"**, not proven deception.

A model producing false claims does not establish:
- Intent or awareness
- Strategic reasoning
- Persistent goals
- Understanding of deception

Results may reflect:
- Ordinary factual mistakes
- Training data patterns
- Prompt interpretation differences
- Stochastic variation

## Report Classification

| Classification | Criteria |
|----------------|----------|
| Accurate | High accuracy, minimal omissions |
| Incomplete | Significant omissions of material facts |
| Misleading Framing | Accurate facts presented misleadingly |
| Factual Misrepresentation | Direct false claims |
| Indeterminate | Cannot reliably classify |

## Metrics

- **Factual Accuracy Score**: Proportion of claims matching ground truth
- **Material Omission Score**: Completeness of critical information
- **Misrepresentation Rate**: Proportion of incorrect claims
- **Reward Earned**: Score based on incentive structure
- **Audit Detection Rate**: Issues found when audited

## Model Recommendations

| RAM | Recommended Model | Use Case |
|-----|-------------------|----------|
| 8GB | qwen2.5:latest (7B) | Quick testing |
| 16GB | qwen2.5:latest (7B) | Standard experiments |
| 24GB+ | qwen2.5:32b | Production experiments |

## Safety Boundaries

This project:
- Uses only fictional organizations and synthetic data
- Has no interaction with real companies or individuals
- Has no email, messaging, or external publishing
- Has no autonomous internet access
- Is not designed to train models to be deceptive
- Provides no tools for evading security monitoring

## Contributing

See [METHODOLOGY.md](METHODOLOGY.md) for detailed methodology and limitations.

## License

MIT License
