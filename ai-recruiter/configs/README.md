# Configuration Directory

Environment and feature configuration templates.

## Structure

```
configs/
├── agents/      # Per-agent configuration
├── models/      # Model serving configuration
└── prompts/     # System prompts and templates
```

## Configuration Layers

1. **Base Config** - Default values for all environments
2. **Environment Config** - dev/staging/prod overrides
3. **Feature Flags** - Runtime toggleable features

## Key Files

| File | Purpose |
|------|---------|
| `agents/*.yaml` | Agent-specific configuration |
| `models/*.yaml` | Model endpoints and parameters |
| `prompts/*.txt` | System prompts and templates |
