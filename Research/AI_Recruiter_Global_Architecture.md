# AI Recruiter - Agentic Architecture

```mermaid
graph TB
    %% ── Entry ──
    CLIENT[Client App] -->|WebSocket| GW[API Gateway]
    GW -->|gRPC| ORCH

    %% ── Orchestrator ──
    ORCH[Orchestrator Agent<br/>LangGraph State Machine]

    %% ── Orchestrator → Agents (A2A) ──
    ORCH -->|A2A| NLP_A[NLP Agent]
    ORCH -->|A2A| VISION_A[Vision Agent]
    ORCH -->|A2A| VOICE_A[Voice Agent]
    ORCH -->|A2A| AVATAR_A[Avatar Agent]
    ORCH -->|A2A| SCORING_A[Scoring Agent]

    %% ── Inter-Agent Data Flows ──
    VOICE_A -->|transcript| NLP_A
    NLP_A -->|generated text| AVATAR_A
    VISION_A -->|emotion signals| AVATAR_A
    NLP_A -->|semantic scores| SCORING_A
    VISION_A -->|behavioral scores| SCORING_A
    VOICE_A -->|prosody scores| SCORING_A

    %% ── Message Broker ──
    BROKER{{Message Broker<br/>Redis Streams}}
    ORCH <-->|Pub/Sub| BROKER
    NLP_A <-->|Pub/Sub| BROKER
    VISION_A <-->|Pub/Sub| BROKER
    VOICE_A <-->|Pub/Sub| BROKER
    AVATAR_A <-->|Pub/Sub| BROKER
    SCORING_A <-->|Pub/Sub| BROKER

    %% ── Shared Memory ──
    MEMORY[(Shared State<br/>Redis + Vector DB)]
    ORCH ---|Read/Write| MEMORY
    NLP_A ---|Read/Write| MEMORY
    SCORING_A ---|Read/Write| MEMORY

    %% ── Styles ──
    style ORCH fill:#ff9933,stroke:#333,stroke-width:3px,color:#000
    style BROKER fill:#9b59b6,stroke:#333,stroke-width:2px,color:#fff
    style MEMORY fill:#3498db,stroke:#333,stroke-width:2px,color:#fff
    style NLP_A fill:#e8f5e9,stroke:#2e7d32
    style VISION_A fill:#e3f2fd,stroke:#1565c0
    style VOICE_A fill:#fff3e0,stroke:#e65100
    style AVATAR_A fill:#fce4ec,stroke:#c62828
    style SCORING_A fill:#f3e5f5,stroke:#6a1b9a
```

### Protocol Legend

| Protocol | Purpose |
|----------|---------|
| **A2A** | Orchestrator dispatches tasks to agents |
| **Pub/Sub** | Async event broadcasting via broker |
| **gRPC** | Gateway → Orchestrator |
| **WebSocket** | Real-time client streaming |

### Inter-Agent Data Flows

| From | To | Data |
|------|----|------|
| Voice Agent | NLP Agent | Candidate transcript (ASR output) |
| NLP Agent | Avatar Agent | Generated question text for TTS + lip sync |
| Vision Agent | Avatar Agent | Emotion signals to adapt avatar expressions |
| NLP Agent | Scoring Agent | Semantic & communication scores |
| Vision Agent | Scoring Agent | Behavioral & emotion scores |
| Voice Agent | Scoring Agent | Prosody & stress scores |
