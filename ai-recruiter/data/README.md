# Data Directory

Canonical data lifecycle zones for the project.

## Structure

```
data/
├── raw/          # Original unmodified data
├── interim/      # Intermediate processing outputs
├── processed/    # Final cleaned datasets
└── exports/      # Generated reports and exports
```

## Data Flow

```
raw/ → interim/ → processed/ → ml/recruiter-persona/training/
                            → exports/ (reports)
```

## data_v2 Integration

The legacy `data_v2/` folder at project root is the current authoritative source.
It maps into this structure as:

- `data_v2/raw/` → `data/raw/recruiter_interviews/`
- `data_v2/final/` → `data/processed/recruiter_persona/`

## Data Governance

- Raw data is immutable - never modify
- All transformations are reproducible via scripts
- Processed data includes provenance metadata
