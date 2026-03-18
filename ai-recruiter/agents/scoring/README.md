# Scoring Agent

Global candidate evaluation and interview assessment engine.

## Purpose

- Aggregate scores from all specialized agents
- Apply weighted scoring models
- Generate interview reports and recommendations
- Track candidate performance across competencies
- Produce final hiring decision support

## Key Components

| File | Purpose |
|------|---------|
| `agent.py` | Main Scoring agent with A2A task handler |
| `score_aggregator.py` | Multi-signal score aggregation |
| `competency_mapper.py` | Map signals to job competencies |
| `weighting_engine.py` | Role-specific scoring weights |
| `report_generator.py` | Interview report generation |
| `recommendation_engine.py` | Hiring recommendation logic |
| `agent_card.json` | A2A discovery metadata |

## Scoring Dimensions

| Source | Metrics |
|--------|---------|
| NLP Agent | Semantic relevance, communication clarity, depth of answers |
| Vision Agent | Confidence, engagement, emotional stability |
| Voice Agent | Vocal confidence, stress levels, speaking pace |

## Data Flows

- **Input from NLP**: Semantic and communication scores
- **Input from Vision**: Behavioral and emotion scores
- **Input from Voice**: Prosody and stress scores
- **Output**: Final evaluation report, recommendation
