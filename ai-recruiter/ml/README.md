# ML Directory

Machine learning workflows for model fine-tuning, evaluation, and serving.

## Structure

```
ml/
├── recruiter-persona/    # Fine-tuning pipeline for Alex (recruiter interview model)
│   ├── data/             # Data preprocessing and formatting
│   ├── training/         # Training scripts and configs
│   ├── evaluation/       # Model evaluation and benchmarks
│   └── serving/          # Inference server and deployment
├── models/               # Model artifacts and checkpoints
└── experiments/          # MLflow/W&B experiment tracking
```

## Primary Pipeline: Recruiter Persona (Alex)

The main ML workflow fine-tunes a conversational model to act as a professional
recruiter named "Alex" using the interview dialogue data from `data_v2`.

### Data Flow
```
data_v2/raw/dataset.csv
    → preprocessing (clean, parse, filter)
    → data_v2/final/*.jsonl (ShareGPT + messages format)
    → training (LoRA/QLoRA fine-tuning)
    → models/recruiter-persona/
    → serving (vLLM / TGI)
```
