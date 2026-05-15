# AI Recruiter — Presentation Diagrams (Mermaid Code)

Use these in any tool that supports Mermaid (PowerPoint plugins, Mermaid Live Editor, draw.io import, etc.).
Render at: https://mermaid.live

---

## 1. Global System Architecture

```mermaid
graph TD
    CANDIDATE["🧑 Candidate Browser<br/>Camera · Microphone · Display"]
    GW["API Gateway"]
    ORCH["🎯 Orchestrator<br/>Interview State Machine"]
    NLP["💬 NLP Agent<br/>Questions & Understanding"]
    VISION["👁️ Vision Agent<br/>Face & Body Analysis"]
    VOICE["🎙️ Voice Agent<br/>Speech & Tone Analysis"]
    AVATAR["🧑‍💼 Avatar Agent<br/>AI Interviewer Face & Voice"]
    SCORE["📊 Scoring Agent<br/>Candidate Evaluation"]
    BROKER["📡 Message Broker<br/>Redis Pub/Sub"]

    CANDIDATE -->|"WebSocket<br/>(real-time)"| GW
    GW -->|gRPC| ORCH

    ORCH -->|A2A| NLP
    ORCH -->|A2A| VISION
    ORCH -->|A2A| VOICE
    ORCH -->|A2A| AVATAR
    ORCH -->|A2A| SCORE

    VOICE -.->|transcript| NLP
    NLP -.->|question text| AVATAR
    VISION -.->|emotion signals| AVATAR

    NLP -.->|semantic scores| SCORE
    VISION -.->|behavioral scores| SCORE
    VOICE -.->|prosody scores| SCORE

    ORCH <-->|events| BROKER
    NLP <-->|events| BROKER
    VISION <-->|events| BROKER
    VOICE <-->|events| BROKER
    AVATAR <-->|events| BROKER
    SCORE <-->|events| BROKER

    style CANDIDATE fill:#f5f5f5,stroke:#333,stroke-width:2px,color:#000
    style GW fill:#eceff1,stroke:#546e7a,stroke-width:2px,color:#000
    style ORCH fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#000
    style NLP fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    style VISION fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    style VOICE fill:#fff8e1,stroke:#f57f17,stroke-width:2px,color:#000
    style AVATAR fill:#fce4ec,stroke:#c62828,stroke-width:2px,color:#000
    style SCORE fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
    style BROKER fill:#ede7f6,stroke:#4527a0,stroke-width:2px,color:#000
```

---

## 2. Orchestrator Interview Flow (State Machine)

```mermaid
stateDiagram-v2
    [*] --> Greeting
    Greeting --> Questioning : Candidate introduced

    Questioning --> FollowUp : Answer is vague
    Questioning --> Questioning : Answer is good, next topic
    FollowUp --> Questioning : Clarified / max follow-ups reached
    FollowUp --> FollowUp : Still vague (≤3 attempts)

    Questioning --> Closing : All topics covered
    Closing --> Scoring : Interview ends
    Scoring --> [*] : Report generated
```

---

## 3. NLP Agent Pipeline (3-LLM Chain)

```mermaid
graph TD
    INPUT["🗣️ Candidate Answer"]
    SCORER["📝 Scorer<br/><i>Qwen 1.5B · CPU</i><br/>Rates answer quality"]
    STATE["🔀 State Tracker<br/>Decides next action"]
    LLM["🤖 Main LLM<br/><i>LLaMA 3.1 8B · GPU</i><br/>Generates next question<br/>+ acknowledges answer"]
    REFINER["✨ Refiner<br/><i>Qwen 1.5B · CPU</i><br/>Sharpens question depth"]
    OUTPUT["💬 To Avatar → Candidate"]

    INPUT --> SCORER
    SCORER --> STATE
    STATE -->|"Follow-up"| LLM
    STATE -->|"New topic"| LLM
    STATE -->|"Close interview"| OUTPUT
    LLM --> REFINER
    REFINER --> OUTPUT

    VAGUE["Vague<br/>< 25 words"] -.-> STATE
    DETAILED["Detailed<br/>≥ 30 words + specifics"] -.-> STATE
    SKIP["Skip requested"] -.-> STATE

    style INPUT fill:#f5f5f5,stroke:#333,stroke-width:2px
    style SCORER fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style STATE fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style LLM fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
    style REFINER fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style OUTPUT fill:#f5f5f5,stroke:#333,stroke-width:2px
    style VAGUE fill:#ffebee,stroke:#c62828,stroke-width:1px
    style DETAILED fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    style SKIP fill:#fff8e1,stroke:#f57f17,stroke-width:1px
```

---

## 4. Vision Agent — Multimodal Stress Detection Pipeline

```mermaid
graph LR
    CAM["📷 Camera<br/>640×480"]
    MIC["🎙️ Microphone<br/>16 kHz"]

    subgraph EXTRACT ["Feature Extraction"]
        FACE["😊 Face Analysis<br/>468 landmarks<br/>8 emotions<br/>gaze tracking"]
        BODY["🏃 Body Analysis<br/>33 pose joints<br/>posture & gestures"]
        AUDIO["🔊 Voice Analysis<br/>Wav2Vec2 embeddings<br/>vocal stress"]
    end

    FUSION["⚙️ Multimodal<br/>Fusion Engine"]

    subgraph PROFILE ["Behavioral Profile"]
        COG["🧠 Cognitive Load"]
        EMO["❤️ Emotional Arousal"]
        ENG["🤝 Engagement"]
        CONF["💪 Confidence"]
    end

    CAM --> FACE
    CAM --> BODY
    MIC --> AUDIO
    FACE --> FUSION
    BODY --> FUSION
    AUDIO --> FUSION
    FUSION --> COG
    FUSION --> EMO
    FUSION --> ENG
    FUSION --> CONF

    style CAM fill:#f5f5f5,stroke:#333,stroke-width:2px
    style MIC fill:#f5f5f5,stroke:#333,stroke-width:2px
    style FACE fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style BODY fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style AUDIO fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    style FUSION fill:#fff3e0,stroke:#e65100,stroke-width:3px
    style COG fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style EMO fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style ENG fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style CONF fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
```

---

## 5. Vision Agent — Detailed Technical Pipeline

```mermaid
graph TD
    FRAME["Video Frame"]
    CHUNK["Audio Chunk"]

    subgraph VISUAL ["Visual Processing"]
        MP_FACE["MediaPipe FaceMesh<br/>468 + iris landmarks"]
        MP_POSE["MediaPipe Pose<br/>33 body joints<br/><i>parallel thread, every 2nd frame</i>"]
        HSEMO["HSEmotion<br/>EfficientNet-B0<br/>8 emotions + valence/arousal"]
        GESTURE["Gesture Analyzer<br/>MediaPipe Hands<br/>fidgeting & self-touch"]
    end

    subgraph AUDITORY ["Audio Processing"]
        WAV2VEC["Wav2Vec2-base<br/>512-D embedding"]
        STUDENT["StudentNet MLP<br/>512→256→2"]
    end

    CALIB["Personal Baseline<br/>45s calibration<br/>Welford's algorithm"]
    ZSCORE["Z-Score<br/>Normalization"]
    MULTI["Multi-Dimension<br/>Weighted Fusion"]
    SESSION["Session Aggregator<br/>Notable moments<br/>PDF report"]

    FRAME --> MP_FACE
    FRAME --> MP_POSE
    FRAME --> HSEMO
    FRAME --> GESTURE
    CHUNK --> WAV2VEC
    WAV2VEC --> STUDENT

    MP_FACE --> ZSCORE
    MP_POSE --> ZSCORE
    HSEMO --> MULTI
    GESTURE --> ZSCORE
    STUDENT --> MULTI
    CALIB --> ZSCORE
    ZSCORE --> MULTI
    MULTI --> SESSION

    style FRAME fill:#f5f5f5,stroke:#333,stroke-width:2px
    style CHUNK fill:#f5f5f5,stroke:#333,stroke-width:2px
    style MP_FACE fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style MP_POSE fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style HSEMO fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style GESTURE fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style WAV2VEC fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    style STUDENT fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    style CALIB fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style ZSCORE fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style MULTI fill:#fff3e0,stroke:#e65100,stroke-width:3px
    style SESSION fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## 6. End-to-End Interview Flow (Sequence)

```mermaid
sequenceDiagram
    participant C as 🧑 Candidate
    participant A as 🧑‍💼 Avatar
    participant O as 🎯 Orchestrator
    participant N as 💬 NLP Agent
    participant Vi as 👁️ Vision Agent
    participant Vo as 🎙️ Voice Agent
    participant S as 📊 Scoring Agent

    O->>A: Start interview greeting
    A->>C: "Hello! Welcome to your interview..."

    loop For Each Question
        O->>N: Request next question
        N->>O: Generated question
        O->>A: Speak question
        A->>C: Asks question (voice + face)

        C->>Vo: Speaks answer (audio stream)
        C->>Vi: Shows face/body (video stream)

        Vo-->>N: Transcript
        Vo-->>S: Prosody & stress scores
        Vi-->>S: Emotion & body language scores
        Vi-->>A: Emotion signals (adapt expression)

        N->>N: Score → Decide → Generate → Refine
        N-->>S: Semantic scores
    end

    O->>S: Request final evaluation
    S->>O: Candidate report
    O->>A: Closing message
    A->>C: "Thank you for your time!"
```

---

## 7. Inter-Agent Data Flow

```mermaid
graph LR
    VOICE["🎙️ Voice Agent"]
    NLP["💬 NLP Agent"]
    VISION["👁️ Vision Agent"]
    AVATAR["🧑‍💼 Avatar Agent"]
    SCORING["📊 Scoring Agent"]

    VOICE -->|"transcript<br/>(what they said)"| NLP
    NLP -->|"question text<br/>(what to say next)"| AVATAR
    VISION -->|"emotion signals<br/>(how they feel)"| AVATAR

    NLP -->|"semantic scores<br/>(answer quality)"| SCORING
    VISION -->|"behavioral scores<br/>(body language)"| SCORING
    VOICE -->|"prosody scores<br/>(voice stress)"| SCORING

    style VOICE fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    style NLP fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style VISION fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style AVATAR fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style SCORING fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
```

---

## Rendering Instructions

**Option A — Mermaid Live Editor:**  
Paste any code block into https://mermaid.live → export as SVG/PNG

**Option B — VS Code Preview:**  
Install "Markdown Preview Mermaid Support" extension → preview this file

**Option C — PowerPoint:**  
Export as SVG from Mermaid Live → Insert as image in slides

**Option D — draw.io:**  
Use Mermaid import plugin or recreate from the structure
