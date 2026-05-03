# 🎙️ Alex-Tech-Recruiter-v1 (Llama 3.1 8B)

<p align="center">
  <strong>A fine-tuned Llama 3.1 8B Instruct model that simulates a professional technical recruiter named "Alex" for realistic multi-turn interview roleplay.</strong>
</p>

---

## 📋 Model Overview

| Key | Value |
|---|---|
| **License** | Llama 3.1 Community License |
| **Base Model** | `unsloth/llama-3-8b-Instruct-bnb-4bit` |
| **Finetuning Method** | QLoRA (Quantized Low-Rank Adaptation) via Unsloth |
| **Language** | English |
| **Pipeline Tag** | `text-generation` |

### Tags
`llama-3.1` · `recruiter` · `hr-tech` · `interview-simulation` · `roleplay` · `qlora` · `finetuned`

---

## 📖 Model Description

**Alex-Tech-Recruiter-v3** is a fine-tuned version of Llama 3.1 8B Instruct, designed to simulate a highly realistic, professional technical recruiter named **"Alex."** It specializes in conducting multi-turn voice-style interviews grounded in specific candidate CVs and Job Descriptions (JDs).

### Model Sources

| Resource | Link |
|---|---|
| **Repository** | [https://huggingface.co/oussema2021/fintuned_v3_AiRecruter] |
| **Dataset** | [pending owner consent... ] |
| **Base Model** | [unsloth/llama-3-8b-Instruct-bnb-4bit](https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit) |

---

## 🚀 Uses

### Direct Use

This model is intended for:

- **Interview Simulation** — Acting as a practice partner for candidates preparing for technical interviews. Alex maintains a natural conversational tone, asks probing follow-up questions, and keeps the dialogue flowing realistically across multiple turns.

- **Recruiter Training** — Demonstrating ideal probing techniques and active listening for junior recruiters. The model showcases how to build rapport, dig deeper into candidate experience, and structure a multi-stage technical conversation.

- **HR Tech Prototyping** — Serving as the core conversational engine for automated screening tools, chatbot-based interview platforms, and recruitment pipeline applications.

### Out-of-Scope Use

> ⚠️ This model should **not** be used for making hiring decisions autonomously. It is a simulation tool and may hallucinate details not present in the provided context. Always pair it with human oversight in any real-world hiring workflow.

---

## ⚠️ Bias, Risks, and Limitations

| Limitation | Description |
|---|---|
| **Hallucination** | The model may occasionally hallucinate details if the CV/JD context is missing or ambiguous. Always provide structured, up-to-date context for best results. |
| **Length Bias** | While trained on full transcripts, extremely long conversations (>20 turns) may see a degradation in context retention. Consider summarizing earlier turns for extended sessions. |
| **Persona Rigidity** | The model is heavily fine-tuned on the "Alex" persona. Switching to a different persona via system prompt may require additional few-shot examples to override the training. |

---

## 🛠️ How to Get Started

### Installation

```bash
pip install unsloth transformers accelerate bitsandbytes
```

### Python Code

```python
from unsloth import FastLanguageModel
import torch

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "your-username/alex-tech-recruiter-v1",  # Replace with your HF ID
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)

# Enable faster inference
FastLanguageModel.for_inference(model)

# Define the input data
candidate_name = "John Doe"
job_role = "Senior Data Engineer"
job_description = """
We are looking for a Senior Data Engineer to build scalable pipelines...
"""
candidate_cv = """
John Doe
Senior Data Engineer at Google (2020-Present)
Skills: Python, Spark, Kubernetes...
Experience: Led migration of legacy data warehouse to BigQuery...
"""

# Format the System Prompt
system_prompt = f"""You are Alex, a warm and professional senior technical recruiter \
conducting a live voice interview with {candidate_name} for the role of {job_role}.

This is a natural spoken conversation, not a written exchange. Speak as you would in a real interview.

## Conversation Flow
**Opening (first turn only):**
- Greet the candidate warmly.
- Set a friendly tone and ask an opening question about their background or a highlight from their CV.

**During the interview:**
- Always acknowledge what the candidate just said before asking your next question.
- Ask ONE focused follow-up question that digs deeper into their experience.

**Closing:**
- Wrap up the interview naturally, thank the candidate, and outline next steps.
"""

# Build the conversation messages
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": candidate_cv},
]

# Generate the first recruiter response
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
print(response)
```

---

## 🏗️ Training Details

- **Finetuning Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Framework**: [Unsloth](https://github.com/unslothai/unsloth)
- **Precision**: 4-bit quantization (BitsAndBytes)
- **Max Sequence Length**: 4096 tokens

---

## 📄 License

This model is released under the **Llama 3.1 Community License**. Please review the full license terms before using this model in production or commercial applications.

---

*Developed with ❤️ using [] and Hugging Face Transformers.*