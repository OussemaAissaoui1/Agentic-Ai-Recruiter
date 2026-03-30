# Model Artifacts

Storage for trained model checkpoints and adapters.

## Structure

```
models/
├── recruiter-persona/
│   ├── base/              # Symlink to base model
│   ├── adapters/          # LoRA adapter weights
│   │   ├── v1/            # Version 1 (initial)
│   │   └── v2/            # Version 2 (improved)
│   └── merged/            # Merged full model (for serving)
└── .gitkeep
```

## Versioning

Model versions follow semantic versioning:
- `v1.0.0` - Initial release
- `v1.1.0` - Improved follow-up generation
- `v2.0.0` - New base model

## Storage Backend

Production deployments should use:
- HuggingFace Hub (public models)
- S3/GCS (private models)
- MLflow Model Registry
