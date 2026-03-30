"""
Pub/Sub Client for Redis Streams

Purpose:
    Wrapper for Redis Streams pub/sub messaging between agents.
    Enables asynchronous event distribution for interview events.

Key Responsibilities:
    - Publish events to Redis Streams
    - Subscribe to event streams with consumer groups
    - Handle message acknowledgment and retries
    - Serialize/deserialize event payloads

Event Streams:
    - interview.transcripts: Voice -> NLP, Scoring
    - interview.questions: NLP -> Avatar
    - interview.emotions: Vision -> Avatar, Scoring
    - interview.scores: All -> Scoring

Usage:
    pubsub = PubSubClient(redis_url)
    await pubsub.publish("interview.transcripts", transcript_event)
    async for event in pubsub.subscribe("interview.questions"):
        process(event)
"""
