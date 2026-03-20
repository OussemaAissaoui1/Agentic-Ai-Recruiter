# Prompt Templates

System prompts and templates for LLM interactions.

## Key Files

| File | Purpose |
|------|---------|
| `recruiter_system.txt` | Main recruiter persona system prompt |
| `question_generation.txt` | Prompt for generating interview questions |
| `response_analysis.txt` | Prompt for analyzing candidate responses |
| `scoring_rubric.txt` | Evaluation criteria prompt |
| `follow_up.txt` | Follow-up question generation prompt |

## Recruiter Persona Prompt

```text
You are Alex, a senior recruiter conducting a structured and professional interview.
Ask one question at a time, adapt follow-up questions to the candidate's last answer,
and keep the tone concise, respectful, and role-relevant.
Avoid discriminatory or illegal questions.
```

## Template Variables

- `{job_role}` - Target position
- `{candidate_name}` - Candidate's name
- `{conversation_history}` - Prior Q&A
- `{competencies}` - Required competencies
