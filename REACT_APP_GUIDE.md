# React Demo App - Quick Start Guide

## 🎯 New Model Configured!

The React app now uses: **`oussema2021/fintuned_v3_AiRecruter`**

---

## 🚀 How to Run the React Demo

### Option 1: Quick Start (Recommended)
```bash
cd ai-recruiter/agents/nlp/demo
./start-react-demo.sh
```

This will:
- ✅ Install npm dependencies (if needed)
- ✅ Start FastAPI backend on port **8000**
- ✅ Start React frontend on port **3000**
- ✅ Show logs in real-time

**Access the app:** http://localhost:3000

### Option 2: Manual Start

#### Terminal 1 - Backend
```bash
cd ai-recruiter/agents/nlp
python3 -m uvicorn demo.server:app --host 0.0.0.0 --port 8000
```

#### Terminal 2 - Frontend
```bash
cd ai-recruiter/agents/nlp/demo
npm install  # First time only
npm run dev
```

**Access the app:** http://localhost:3000

---

## 📊 Features Enabled

### Main Model (GPU)
- **Model:** `oussema2021/fintuned_v3_AiRecruter`
- **Size:** 8B parameters
- **Device:** CUDA GPU
- **Speed:** ~500ms per question

### Answer Scorer (CPU)
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Purpose:** Analyzes candidate answers for quality/engagement
- **Device:** CPU (runs in parallel)
- **Speed:** ~400-800ms

### Question Refiner (CPU)
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Purpose:** Enhances questions with CV-specific details
- **Device:** CPU (runs in parallel)
- **Speed:** ~400-900ms
- **Features:**
  - Preserves greetings/introductions
  - Adds acknowledgments to questions
  - Makes questions more specific

### Text-to-Speech
- **Voice:** af_heart
- **Format:** WAV audio
- **Streaming:** Real-time chunk delivery

---

## 📁 Architecture

```
React Frontend (Port 3000)
    ↓ HTTP POST /api/ask
FastAPI Backend (Port 8000)
    ↓
NLPAgent (server.py)
    ├─→ Main Model (GPU) - Question Generation
    ├─→ Scorer (CPU) - Answer Analysis
    ├─→ Refiner (CPU) - Question Enhancement
    └─→ TTS (CPU) - Voice Synthesis
```

---

## 🔍 Monitoring Logs
### View Real-time Logs
```bash
# Backend logs (model outputs, scorer, refiner)
tail -f ai-recruiter/agents/nlp/demo/backend.log
# Frontend logs (React dev server)
tail -f ai-recruiter/agents/nlp/demo/frontend.log
```
### Search Logs
```bash
cd ai-recruiter/agents/nlp/demo

# View scorer analysis
grep "SCORER" backend.log

# View question refinements
grep "REFINER" backend.log

# View final questions
grep "FINAL" backend.log
```

---

## 🎮 Using the App

1. **Enter Candidate CV** (or use the default)
2. **Click "Start Interview"**
3. **Listen** to the recruiter's greeting (with TTS)
4. **Type** candidate's answer
5. **Send** - the system will:
   - Analyze the answer quality (Scorer)
   - Generate next question (Main Model)
   - Refine the question (Refiner)
   - Convert to speech (TTS)
6. **Repeat** the conversation

---

## 🛠 Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is in use
lsof -ti:8000 | xargs kill -9

# Try manual start with more verbose output
cd ai-recruiter/agents/nlp
python3 -m uvicorn demo.server:app --reload
```

### Frontend won't start
```bash
# Reinstall dependencies
cd ai-recruiter/agents/nlp/demo
rm -rf node_modules package-lock.json
npm install

# Try manual start
npm run dev
```

### Model loading errors
- First run downloads ~16GB from HuggingFace
- Ensure you have **8-10GB GPU memory** available
- Check CUDA is available: `python3 -c "import torch; print(torch.cuda.is_available())"`

### Scorer/Refiner not working
- These run on CPU, so slower machines may timeout
- Check logs: `grep "scorer\|refiner" backend.log`
- Disable if needed: Edit `server.py` lines 74-77

---

## 📦 Configuration

Edit `ai-recruiter/agents/nlp/demo/server.py` to customize:

```python
_agent = NLPAgent(
    # Change model
    model_path="your/model/path",
    
    # Disable scorer/refiner for speed
    enable_scorer=False,
    enable_refiner=False,
    
    # Change TTS voice
    tts_voice="af_bella",  # or af_nicole, af_sarah, etc.
    
    # Adjust GPU memory
    gpu_memory_utilization=0.60,  # 0.0-1.0
)
```

---

## 🎯 Performance Tips

### For Best Quality (Slower)
- ✅ Enable scorer
- ✅ Enable refiner
- ✅ Enable TTS
- 📊 Total: ~2-3 seconds per turn

### For Speed (Faster)
- ❌ Disable scorer
- ❌ Disable refiner
- ❌ Disable TTS
- 📊 Total: ~500ms per turn

### Recommended (Balanced)
- ✅ Enable scorer
- ✅ Enable refiner
- ❌ Disable TTS (or let user toggle)
- 📊 Total: ~1-2 seconds per turn

---

## 📝 Next Steps

1. ✅ Model updated to `fintuned_v3_AiRecruter`
2. ✅ Scorer/Refiner enabled in server
3. 🔄 Run `./start-react-demo.sh`
4. 🔄 Test the interview flow
5. 🔄 Check logs for scorer/refiner output

Enjoy your AI Recruiter! 🎉
