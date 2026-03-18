# ML Documentation

Machine learning pipeline documentation.

## Key Documents

| Document | Purpose |
|----------|---------|
| `training_guide.md` | How to train the recruiter persona model |
| `data_preparation.md` | Data preprocessing pipeline details |
| `evaluation_guide.md` | Model evaluation methodology |
| `serving_guide.md` | Model deployment and serving |
| `experiments.md` | Experiment tracking with MLflow |

## Quick Start

1. Prepare data: `python scripts/data/process_recruiter_data.py`
2. Train model: `python ml/recruiter-persona/training/train.py`
3. Evaluate: `python ml/recruiter-persona/evaluation/evaluate.py`
4. Serve: `python ml/recruiter-persona/serving/server.py`
