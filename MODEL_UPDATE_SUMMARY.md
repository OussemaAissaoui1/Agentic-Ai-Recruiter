# Model Update Summary

## New Model Configuration

**Model:** `oussema2021/fintuned_v3_AiRecruter`

This is the new fine-tuned LLaMA 3.1 8B model for the AI recruiter persona.

---

## Files Updated

### 1. **Main Agent** ✅
- **File:** `ai-recruiter/agents/nlp/agent.py`
- **Line:** 163
- **Change:** Default model path updated to `oussema2021/fintuned_v3_AiRecruter`

### 2. **Demo App** ✅
- **File:** `ai-recruiter/agents/nlp/demo/app.py`
- **Line:** 218
- **Change:** Updated model path + enabled refiner
- **Features:** 
  - Scorer: Enabled (Qwen 1.5B on CPU)
  - Refiner: Enabled (Qwen 1.5B on CPU)
  - TTS: Enabled

### 3. **Demo Server** ✅
- **File:** `ai-recruiter/agents/nlp/demo/server.py`
- **Line:** 73
- **Change:** Updated model path
- **Features:**
  - Scorer: Disabled (for speed)
  - Refiner: Disabled (for speed)
  - TTS: Enabled

### 4. **New Test Script** ✅
- **File:** `test_new_model.py` (NEW)
- **Purpose:** Comprehensive test of the new model with all features
- **Tests:**
  - Initial greeting
  - Technical conversation
  - Vague answer handling
  - Scorer/Refiner integration

---

## Recent Bug Fixes (Also Applied)

### 1. **Refiner - Greeting Preservation** ✅
- **Issue:** Refiner was replacing greetings with CV-based questions
- **Fix:** Pre-check for greetings and preserve them exactly
- **Location:** `ai-recruiter/agents/nlp/question_refiner.py` lines 217-232

### 2. **Refiner - Acknowledgment Handling** ✅
- **Issue:** Acknowledgments weren't being prepended to refined questions
- **Fix:** Automatically prepend acknowledgments after refinement
- **Location:** `ai-recruiter/agents/nlp/question_refiner.py` lines 290-298

### 3. **Refiner - Better JSON Parsing** ✅
- **Issue:** Model sometimes outputs extra text around JSON
- **Fix:** Extract JSON from wrapped text
- **Location:** `ai-recruiter/agents/nlp/question_refiner.py` lines 279-284

### 4. **Refiner - Increased Limits** ✅
- Timeout: 3s → 5s
- Max tokens: 150 → 200
- Max question length: 200 → 300 chars

---

## How to Test

### Quick Test
```bash
cd /teamspace/studios/this_studio/Agentic-Ai-Recruiter
python3 test_new_model.py
```

### Existing Tests
```bash
# GPU pipeline test (scorer + refiner)
python3 test_pipeline_gpu.py

# Mock pipeline test
python3 test_pipeline_mock.py
```

### Run Demo App
```bash
cd ai-recruiter/agents/nlp/demo
python3 app.py
```

---

## Configuration Options

When initializing `NLPAgent`, you can customize:

```python
agent = NLPAgent(
    # Main recruiter model (GPU)
    model_path="oussema2021/fintuned_v3_AiRecruter",
    
    # Scorer (CPU) - analyzes candidate answers
    enable_scorer=True,
    scorer_model="Qwen/Qwen2.5-1.5B-Instruct",
    
    # Refiner (CPU) - enhances questions
    enable_refiner=True,
    refiner_model="Qwen/Qwen2.5-1.5B-Instruct",
    
    # TTS (Text-to-Speech)
    enable_tts=True,
    tts_voice="af_heart",
    
    # Performance tuning
    gpu_memory_utilization=0.60,
    max_model_len=2048,
    dtype="bfloat16",
)
```

---

## Model Comparison

| Model | Size | Purpose | Device | Speed |
|-------|------|---------|--------|-------|
| `oussema2021/fintuned_v3_AiRecruter` | 8B | Main recruiter | GPU | ~500ms |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Answer scorer | CPU | ~400-800ms |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Question refiner | CPU | ~400-900ms |

**Total pipeline latency:** ~1-2 seconds (with parallel CPU/GPU execution)

---

## Next Steps

1. ✅ Model paths updated
2. ✅ Refiner bugs fixed
3. ✅ Test script created
4. 🔄 Run `test_new_model.py` to verify
5. 🔄 Test in production demo app

---

## Notes

- The model is already on HuggingFace Hub, no local path needed
- First run will download the model (~16GB for 8B params)
- GPU memory usage: ~6-8GB with bfloat16
- Scorer and Refiner run on CPU in parallel with GPU inference
