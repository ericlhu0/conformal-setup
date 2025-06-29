# Conformal Prediction Setup

![CI Check](https://github.com/ericlhu0/conformal-setup/actions/workflows/ci.yml/badge.svg)

Setup to run different conformal prediction experiments.

## Philosophy

Many research projects involve developing **approaches** that perform well on some **benchmarks** with respect to some **metrics**. See [this blog post](https://lis.csail.mit.edu/whats-your-problem-an-oft-missing-section-in-ai-papers/) for an elaboration. This code is organized accordingly:

1. A [base benchmark](src/conformal_setup/benchmarks/base_benchmark.py) class defines the interface for all benchmarks.
2. A [base approach](src/conformal_setup/approaches/base_approach.py) class defines the interface for all approaches.
3. An [experiment script](experiments/run_single_experiment.py) evaluates approaches on benchmarks while recording metrics of interest.

## Example Use

We use [hydra](https://hydra.cc/) for experiments. The [conf](experiments/conf/) directory defines configurations. To run a single approach on a single benchmark with a single random seed:

```
python experiments/run_single_experiment.py seed=42 approach=random benchmark=maze_small
```

To run multiple approaches on multiple benchmarks with multiple random seeds:

```
python experiments/run_single_experiment.py -m seed=1,2,3 approach=random benchmark=maze_small,maze_large
```

Hydra also works well with clusters. For example, see [here](https://hydra.cc/docs/plugins/submitit_launcher/).

## Setup

### OpenAI API Configuration

This project includes OpenAI-based models for conformal prediction. To use these models:

1. **Get an OpenAI API key**: Sign up at [OpenAI](https://platform.openai.com/) and create an API key
2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   # Optional: if you have an organization
   export OPENAI_ORG_ID="your-org-id-here"
   ```
3. **Or create a `.env` file** in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   OPENAI_ORG_ID=your-org-id-here
   ```

### Installation

Install the package with dependencies:
```bash
pip install -e .
```

For development:
```bash
pip install -e ".[develop]"
```

The OpenAI models automatically cache responses to minimize API costs and improve performance. Cache files are stored in `.cache/openai/` by default.

