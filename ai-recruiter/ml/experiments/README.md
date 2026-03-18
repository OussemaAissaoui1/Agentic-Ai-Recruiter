# Experiments

MLflow/Weights & Biases experiment tracking.

## Purpose

Track training runs, hyperparameters, and model artifacts for reproducibility.

## Structure

```
experiments/
├── mlflow.db            # Local MLflow tracking database
├── runs/                # MLflow run artifacts
└── configs/             # Experiment configurations
```

## Tracked Metrics

- Training loss and eval loss
- Perplexity
- Custom interview quality metrics
- Safety benchmark scores

## Usage

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db

# Log a training run
python ml/recruiter-persona/training/train.py --experiment-name recruiter-v2
```
