# AI Recruiter — Presentation Slides Description

---

## SLIDE 1: Title Slide

**Title:** AI-Powered Agentic Interview Platform  
**Subtitle:** A Multimodal, Multi-Agent System for Automated Candidate Assessment  
**Visuals:** Clean project logo / abstract visual of an AI avatar facing a candidate silhouette, connected by dotted lines representing voice, vision, and language signals.

---

## SLIDE 2: Introduction

**Title:** What is the AI Recruiter?

**Bullet points:**
- An intelligent interview platform that **conducts, analyzes, and scores** job interviews autonomously
- Combines **5 specialized AI agents**: language understanding, computer vision, voice analysis, avatar generation, and scoring
- Agents collaborate in real time through a centralized orchestrator — like a team of expert interviewers behind the scenes
- The candidate sees and talks to a single **AI-generated avatar** — natural, conversational, and adaptive

**Speaker note:** Frame this as "replacing the repetitive first-round HR screening, not replacing HR professionals."

---

## SLIDE 3: Problem Statement

**Title:** The Recruitment Bottleneck

**Left column — The Numbers:**
- Average time-to-hire: **36–44 days** globally
- HR teams spend **~23 hours** screening for a single hire
- **75%** of applicants are unqualified — HR still reviews them manually
- First-round interviews are **repetitive, subjective, and hard to scale**

**Right column — The Pain Points:**
- **Inconsistency** — Different interviewers, different standards, different moods
- **Unconscious bias** — Name, appearance, accent influence outcomes
- **Candidate experience** — Scheduling delays, ghosting, impersonal mass rejection
- **No data-driven insight** — Gut feeling replaces structured evaluation

**Visual:** A funnel graphic: 500 applications → 50 screened → 10 interviewed → 1 hired — highlighting the manual bottleneck at the screening/interview stage.

---

## SLIDE 4: Market Study

**Title:** The AI Recruitment Market

**Market size block:**
- Global AI-in-recruitment market: **$590M (2023) → $1.1B+ by 2028** (CAGR ~14%)
- Video interview platforms alone: **$300M+** and growing post-COVID
- **67%** of hiring managers say AI saves time; **43%** cite reduced bias

**Competitive landscape — simplified 2×2 matrix:**

|                        | Text/Resume Only | Multimodal (Voice + Video + NLP) |
|------------------------|:-:|:-:|
| **Rule-Based / Simple** | ATS systems (Greenhouse, Lever) | Basic video platforms (Spark Hire) |
| **AI-Driven / Adaptive** | Chatbot screeners (Olivia, Paradox) | **Our system** ← GAP |

**Key insight (bold callout):**
> Existing tools either do **one modality well** (NLP chatbots, video recording platforms) or do **light AI** on multiple signals. No current platform combines **real-time adaptive NLP + live facial/body emotion analysis + voice stress detection + AI avatar delivery** in one integrated system.

**Speaker note:** Emphasize the white-space opportunity — we're not competing with ATS tools, we're occupying an unserved intersection.

---

## SLIDE 5: Proposed Solution

**Title:** An Agentic AI Interview System

**Core concept (centered, large text):**
> A team of **5 specialized AI agents** that collaborate in real time to conduct, analyze, and score interviews — coordinated by a central orchestrator.

**Agent cards (visual: 5 colored cards in a row):**

| Agent | What It Does | Key Tech |
|-------|-------------|----------|
| **NLP Agent** | Asks adaptive questions, understands answers, decides follow-ups | LLaMA 3.1 8B (fine-tuned recruiter persona) + Qwen scorer + Qwen refiner |
| **Vision Agent** | Reads facial expressions, body language, eye contact, stress signals | HSEmotion, MediaPipe Face+Pose, multi-dimensional behavioral profiling |
| **Voice Agent** | Transcribes speech, analyzes tone, detects vocal stress | Whisper (ASR), Wav2Vec2 + StudentNet (stress), prosody analysis |
| **Avatar Agent** | Generates a realistic AI interviewer face and voice | TTS, Wav2Lip/SadTalker, First Order Motion Model |
| **Scoring Agent** | Aggregates all signals into a fair, structured candidate score | Multi-source weighted fusion |

**Bottom tagline:**  
"Each agent is an expert in its domain. The orchestrator makes them work as one."

---

## SLIDE 6: Architecture — Global System Overview

**Title:** System Architecture

**Diagram (simplified, clean, to be created as a visual):**

```
┌─────────────────────────────────────────────────────────────┐
│                      CANDIDATE BROWSER                      │
│            Camera + Microphone + Video Display               │
└──────────────────────┬──────────────────────────────────────┘
                       │  WebSocket (real-time)
                       ▼
              ┌─────────────────┐
              │   API Gateway   │
              └────────┬────────┘
                       │  gRPC
                       ▼
        ┌──────────────────────────────┐
        │     ORCHESTRATOR AGENT       │
        │   (Central State Machine)    │
        │                              │
        │  Greeting → Questions →      │
        │  Follow-ups → Closing →      │
        │  Scoring                     │
        └──┬─────┬──────┬──────┬──────┬┘
           │     │      │      │      │
     ┌─────┘  ┌──┘   ┌──┘   ┌─┘    ┌─┘
     ▼        ▼      ▼      ▼      ▼
  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
  │ NLP  │ │Vision│ │Voice │ │Avatar│ │Score │
  │Agent │ │Agent │ │Agent │ │Agent │ │Agent │
  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──────┘
     │        │        │        │
     └────────┴────┬───┴────────┘
                   ▼
          ┌────────────────┐
          │ Message Broker │
          │ (Redis Pub/Sub)│
          └────────────────┘
```

**Key callouts (3 brief notes alongside the diagram):**
1. **Agents are independent microservices** — each can be developed, tested, and scaled separately
2. **Orchestrator drives the interview flow** — a state machine that moves through greeting → questioning → follow-ups → closing → scoring
3. **All agents share context** through a message broker and shared memory — so the avatar knows the emotion state, the scorer knows the transcript

---

## SLIDE 7: Architecture — Vision Agent Pipeline (Deep Dive)

**Title:** Vision Agent — How We Read Body Language

**Diagram (simplified, clean flow — left to right):**

```
  CAMERA FEED                    FEATURE EXTRACTION                ANALYSIS                    OUTPUT
 ┌──────────┐     ┌──────────────────────────────────┐     ┌─────────────────┐     ┌──────────────────┐
 │  Webcam   │     │  Face Analysis                   │     │                 │     │  4-D Behavioral  │
 │  640×480  │────▶│  · 468 facial landmarks          │────▶│   Multimodal    │────▶│  Profile:        │
 │  frames   │     │  · Eye gaze tracking             │     │   Fusion        │     │                  │
 └──────────┘     │  · 8 emotion classes + valence    │     │   Engine        │     │  · Cognitive Load│
                   │                                    │     │                 │     │  · Emotional     │
 ┌──────────┐     │  Body Analysis                    │     │  Combines all   │     │    Arousal       │
 │  Micro-   │     │  · 33 body pose joints            │────▶│  signals with   │     │  · Engagement    │
 │  phone    │────▶│  · Posture, gestures, fidgeting   │     │  weighted       │     │    Level         │
 │  16kHz    │     │                                    │     │  scoring        │     │  · Confidence    │
 └──────────┘     │  Voice Analysis                   │     │                 │     │    Level         │
                   │  · Wav2Vec2 audio embeddings       │────▶│                 │     │                  │
                   │  · Vocal stress detection          │     │                 │     │  + Stress Score  │
                   └──────────────────────────────────┘     └─────────────────┘     └──────────────────┘
```

**Below the diagram — 3 highlight boxes:**

| Personal Baseline Calibration | Multi-Dimensional Scoring | Ethical Design |
|:--:|:--:|:--:|
| 45-second calibration per candidate. All scores are relative to **their own** baseline — not compared to others. | 4 behavioral dimensions replace a single stress number. Richer, fairer, more actionable. | HR-friendly language ("Engagement Level" not "Disengagement Score"). Disclaimer on every report. |

---

## SLIDE 8: Architecture — NLP Agent Pipeline (Deep Dive)

**Title:** NLP Agent — How We Conduct the Conversation

**Diagram (top-to-bottom flow):**

```
  CANDIDATE ANSWER
        │
        ▼
  ┌─────────────┐
  │   SCORER    │  "How good was that answer?"
  │  (Qwen 1.5B)│  → Vague / Adequate / Detailed
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │   STATE     │  Decides next action:
  │  TRACKER    │  → Follow-up? New topic? Close?
  └──────┬──────┘
         │
         ▼
  ┌─────────────────┐
  │   MAIN LLM      │  Generates the next question
  │  (LLaMA 3.1 8B) │  CV-aware, context-aware
  │  Recruiter       │  Acknowledges previous answer
  │  Persona         │
  └──────┬──────────┘
         │
         ▼
  ┌─────────────┐
  │   REFINER   │  "Make it deeper and more specific"
  │  (Qwen 1.5B)│  → Final polished question
  └──────┬──────┘
         │
         ▼
    TO CANDIDATE
    (via Avatar)
```

**Key callouts:**
- **3-LLM pipeline**: Scorer evaluates → Main model generates → Refiner sharpens
- **Adaptive**: Vague answers get follow-ups; detailed answers advance to new topics
- **CV-grounded**: Questions reference the candidate's actual experience
- **Duplicate-proof**: Tracks all asked questions to avoid repetition

---

## SLIDE 9: Progress So Far

**Title:** What We've Built

**Visual: A progress tracker — 6 modules, each with a status indicator**

| Module | Status | Details |
|--------|--------|---------|
| **NLP Agent** | ✅ Operational | 3-LLM pipeline (Main + Scorer + Refiner). Adaptive questioning, CV-aware prompts, memory fix applied. FastAPI server running. |
| **Vision Agent** | 🔨 In Development | Multimodal stress detection pipeline complete: HSEmotion (face), MediaPipe (body pose), Wav2Vec2 (audio stress). React frontend + FastAPI WebSocket server. 4-dimensional behavioral profiling designed. Session reports with PDF generation. |
| **Voice Agent** | 📋 Designed | Architecture defined. Whisper ASR, prosody analyzer, VAD, stress detector components spec'd. |
| **Avatar Agent** | 📋 Designed | Architecture defined. TTS + lip-sync + face generation pipeline spec'd. |
| **Orchestrator** | 📋 Designed | LangGraph state machine defined. Nodes, edges, state schema written. |
| **Scoring Agent** | 📋 Designed | Multi-source aggregation architecture defined. |

**Bottom summary bar:**  
> **2 agents operational / in-dev** · **4 agents designed** · **Core infrastructure (A2A, Pub/Sub, shared schemas) scaffolded**

---

## SLIDE 10: Next Steps

**Title:** Roadmap — What Comes Next

**Timeline visual (horizontal, 3 phases):**

**Phase 1 — Complete Core Agents**
- Finish Vision Agent: train visual MLP on StressID dataset, evaluate fusion strategies, latency benchmarking (<200ms target)
- Build Voice Agent: integrate Whisper ASR, prosody analysis, connect to NLP agent
- Build Avatar Agent: TTS + lip-sync pipeline, expression adaptation from emotion signals

**Phase 2 — Integration & Orchestration**
- Wire all agents through the Orchestrator state machine
- Implement Redis message broker for Pub/Sub communication
- End-to-end interview flow: greeting → questions → follow-ups → closing → scoring
- Build the Scoring Agent: aggregate NLP + Vision + Voice signals into structured candidate report

**Phase 3 — Production Readiness**
- Candidate and HR-facing web dashboards
- Consent and ethics review before any pilot deployment
- Noise/lighting robustness testing for real-world environments
- Cross-language audio evaluation
- Scalability testing (concurrent interviews)
- Security audit and data privacy compliance

**Bottom callout (bold):**
> Immediate priority: **Vision Agent completion** + **Voice Agent build** → first full multimodal interview demo

---

## SLIDE 11: Closing / Thank You

**Title:** Thank You  
**Subtitle:** Questions?  
**Visual:** The system architecture diagram (simplified version from Slide 6) as a faded background.  
**Contact/team info placeholder.**

---

## NOTES FOR SLIDE DESIGN

**General style guidance:**
- Clean, minimal design — white/light backgrounds, one accent color per agent (consistent across slides)
- All diagrams should use simple rounded boxes and straight arrows — no 3D, no gradients, no clip art
- Fonts: sans-serif, large (24pt+ body, 32pt+ titles)
- Diagrams: use the ASCII layouts above as structure guides, but render them as proper vector graphics in the slide tool
- Color code agents consistently: NLP=green, Vision=blue, Voice=orange, Avatar=pink, Scoring=purple, Orchestrator=gold
- Keep text sparse on slides — the descriptions above include speaker notes for verbal delivery
