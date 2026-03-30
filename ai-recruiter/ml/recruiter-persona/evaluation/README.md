# Model Evaluation

Evaluation scripts and benchmarks for the fine-tuned recruiter model.

## Key Files

| File | Purpose |
|------|---------|
| `evaluate.py` | Main evaluation entrypoint |
| `metrics.py` | Custom metrics (perplexity, BLEU, interview-specific) |
| `benchmarks.py` | Benchmark suite definitions |
| `human_eval.py` | Human evaluation framework |
| `safety_check.py` | Check for illegal/sensitive question generation |

## Evaluation Dimensions

1. **Fluency** - Language quality and coherence
2. **Relevance** - Questions appropriate to job role
3. **Professionalism** - Maintains recruiter persona
4. **Safety** - No illegal/discriminatory questions
5. **Adaptiveness** - Follow-ups based on candidate responses

## Benchmark Datasets

- Hold-out eval split from data_v2
- Synthetic edge cases
- Adversarial prompts
