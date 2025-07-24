# Safe Feedback Interpretation

![CI Check](https://github.com/ericlhu0/safe-feedback-interpretation/actions/workflows/ci.yml/badge.svg)

## Setup

### OpenAI API Configuration

This project includes OpenAI-based models. To use these models:

1. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   # Optional: if you have an organization
   export OPENAI_ORG_ID="your-org-id-here"
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

### Run Experiments

Checking model confidence with input contradictions across modalities
```bash
python experiments/modality_disagreement.py -m prompt=a,b,c expression_input=img,txt
```

### Thanks
Thanks to Tom Silver for this repo template! https://github.com/tomsilver/python-research-starter