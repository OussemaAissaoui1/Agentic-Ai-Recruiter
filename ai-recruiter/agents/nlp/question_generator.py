"""
Question Generator

Purpose:
    Generates contextual interview questions using the fine-tuned
    recruiter persona model (Alex).

Key Features:
    - Role-specific question generation (based on job_role)
    - Follow-up questions based on candidate responses
    - Behavioral and situational question templates
    - Avoids illegal/discriminatory questions

Integration:
    - Uses recruiter-persona model from ml/recruiter-persona/serving/
    - Applies prompt templates from configs/prompts/
    - Considers conversation history for context

Usage:
    generator = QuestionGenerator(model_client)
    question = await generator.generate(
        job_role="Software Engineer",
        conversation_history=messages,
        phase="QUESTIONING"
    )
"""
