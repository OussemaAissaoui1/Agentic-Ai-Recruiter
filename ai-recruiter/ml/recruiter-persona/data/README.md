# Data Processing for Recruiter Persona

Scripts to prepare interview dialogue data for fine-tuning.

## Purpose

Transform raw interview transcripts into model-ready formats:
- Clean and normalize dialogue turns
- Parse speaker roles (interviewer vs candidate)
- Filter quality conversations
- Remove sensitive/illegal questions
- Output ShareGPT and messages JSONL formats

## Key Files

| File | Purpose |
|------|---------|
| `preprocess.py` | Main preprocessing pipeline (adapted from data_v2) |
| `validators.py` | Conversation quality validators |
| `formatters.py` | Output formatters (ShareGPT, messages, Alpaca) |
| `deduplicator.py` | Exact and near-duplicate removal |
| `splitter.py` | Stratified train/eval splitting |

## Source Data

References `data_v2/` as the authoritative source:
- `data_v2/raw/dataset.csv` - Raw interview dialogues
- `data_v2/final/*.jsonl` - Processed outputs
