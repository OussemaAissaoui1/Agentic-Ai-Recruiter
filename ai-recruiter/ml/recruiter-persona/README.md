# Recruiter Persona Fine-Tuning

Fine-tuning pipeline for the **Alex** recruiter persona - the conversational AI
that conducts professional interviews.

## Pipeline Overview

1. **Data Preparation** (`data/`) - Process data_v2 interview dialogues
2. **Training** (`training/`) - LoRA/QLoRA fine-tuning on base model
3. **Evaluation** (`evaluation/`) - Quality metrics and human eval
4. **Serving** (`serving/`) - Deploy for inference

## Model Identity

- **Name**: Alex
- **Role**: Senior recruiter conducting structured professional interviews
- **Style**: Concise, respectful, role-relevant, one question at a time
- **Trained on**: data_v2 recruiter dialogue dataset

## Training Configuration

- Base Model: Llama 3.x / Mistral
- Method: QLoRA (4-bit quantization + LoRA adapters)
- Format: ShareGPT / Messages JSONL
- Train/Eval Split: 90/10 stratified by job role
