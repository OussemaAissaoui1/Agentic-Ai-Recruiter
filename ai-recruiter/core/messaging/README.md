# Messaging Module

Redis Streams pub/sub for async event distribution.

## Purpose

Enable asynchronous communication between agents:
- Publish interview events (question asked, answer received)
- Subscribe to relevant event streams
- Decouple agent execution

## Key Files

| File | Purpose |
|------|---------|
| `broker.py` | Redis Streams broker abstraction |
| `publisher.py` | Event publishing utilities |
| `consumer.py` | Stream consumer with consumer groups |
| `events.py` | Event type definitions |
| `serializers.py` | Event serialization (JSON/Protobuf) |

## Event Streams

| Stream | Publishers | Subscribers |
|--------|------------|-------------|
| `interview.transcripts` | Voice | NLP, Scoring |
| `interview.questions` | NLP | Avatar |
| `interview.emotions` | Vision | Avatar, Scoring |
| `interview.scores` | All agents | Scoring |
