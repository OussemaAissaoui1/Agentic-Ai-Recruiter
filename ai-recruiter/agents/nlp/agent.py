"""
NLP Agent - Conversation and Question Generation

Purpose:
    Handles all natural language processing tasks for the interview.
    Uses the fine-tuned recruiter persona model (Alex) to generate
    professional interview questions.

Key Responsibilities:
    - Generate contextual interview questions based on job role
    - Analyze candidate responses for semantic content
    - Extract communication quality signals
    - Produce scores for the Scoring Agent

Model Integration:
    Calls the recruiter-persona model served via ml/recruiter-persona/serving/
    using the prompt templates from configs/prompts/

A2A Tasks Handled:
    - generate_question: Create next interview question
    - analyze_response: Semantic analysis of candidate answer
    - extract_scores: Communication quality metrics
"""
