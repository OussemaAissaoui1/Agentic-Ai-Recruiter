# Vision Agent

Computer vision agent for non-verbal communication analysis.

## Purpose

- Analyze facial expressions and micro-expressions
- Detect emotional states (confidence, nervousness, engagement)
- Monitor posture and body language cues
- Track eye contact and attention signals
- Generate behavioral scores for evaluation

## Key Components

| File | Purpose |
|------|---------|
| `agent.py` | Main Vision agent with A2A task handler |
| `face_detector.py` | Face detection and landmark extraction |
| `emotion_classifier.py` | Facial emotion recognition model |
| `posture_analyzer.py` | Body posture and gesture analysis |
| `attention_tracker.py` | Eye gaze and attention monitoring |
| `behavioral_scorer.py` | Aggregate behavioral metrics |
| `agent_card.json` | A2A discovery metadata |

## Models Used

- Face detection: MediaPipe / MTCNN
- Emotion recognition: Custom CNN or FER model
- Posture: BlazePose / OpenPose

## Data Flows

- **Input**: Video frames from client stream
- **Output to Avatar**: Emotion signals for adaptive expressions
- **Output to Scoring**: Behavioral and emotion scores
