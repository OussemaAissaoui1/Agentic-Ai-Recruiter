# NLP Agent

Handles all natural language processing for the interview system.

## Purpose

- Generate contextual interview questions using fine-tuned recruiter model
- Analyze candidate responses for semantic content
- Extract communication quality signals (clarity, relevance, depth)
- Produce semantic scores for the Scoring Agent

## Key Components

| File | Purpose |
|------|---------|
| `agent.py` | Main NLP agent with A2A task handler |
| `question_generator.py` | Interview question generation using recruiter persona LLM |
| `response_analyzer.py` | Semantic analysis of candidate responses |
| `communication_scorer.py` | Communication quality metrics extraction |
| `prompts.py` | Prompt templates for different interview phases |
| `tools.py` | LangChain tools for NLP operations |
| `agent_card.json` | A2A discovery metadata |

## Model Integration

Uses the fine-tuned **recruiter persona model** (Alex) from `ml/recruiter-persona/`.
The model is trained on data_v2 to maintain professional interview style.

## Data Flows

- **Input**: Transcript from Voice Agent
- **Output to Avatar**: Generated question text for TTS + lip sync
- **Output to Scoring**: Semantic and communication scores
