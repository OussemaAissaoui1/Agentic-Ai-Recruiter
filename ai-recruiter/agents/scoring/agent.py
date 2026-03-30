"""
Scoring Agent - Candidate Evaluation Engine

Purpose:
    Aggregates scores from all agents and produces the final
    interview evaluation and hiring recommendation.

Key Responsibilities:
    - Collect scores from NLP, Vision, Voice agents
    - Apply weighted scoring models per competency
    - Map signals to job-specific competencies
    - Generate interview reports
    - Produce hiring recommendations

Scoring Dimensions:
    - NLP: Semantic relevance, communication clarity, depth
    - Vision: Confidence, engagement, emotional stability
    - Voice: Vocal confidence, stress levels, pace

A2A Tasks Handled:
    - record_score: Store incoming score from agent
    - compute_final: Calculate weighted final scores
    - generate_report: Create interview evaluation report
"""
