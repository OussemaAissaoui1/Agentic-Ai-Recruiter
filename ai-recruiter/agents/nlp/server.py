"""
NLP Agent FastAPI Server

RESTful API for the AI Recruiter NLP Agent.

Endpoints:
    POST /generate-question    - Generate next interview question
    POST /analyze-response     - Analyze candidate response
    GET  /session/{id}/stats   - Get session statistics
    POST /session/{id}/reset   - Reset interview session
    DELETE /session/{id}       - End interview session
    GET  /health               - Health check
    POST /warmup               - Warm up the model
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict, Any
import logging
import os

from agent import NLPAgent, QuestionGenerationRequest, QuestionGenerationResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="NLP Agent - AI Recruiter",
    description="Natural Language Processing agent for AI-powered recruitment interviews",
    version="1.0.0",
)

# Initialize NLP Agent (lazy loaded on first request)
nlp_agent: Optional[NLPAgent] = None


# ============================================================================
# Pydantic Models for API
# ============================================================================

class ConversationTurn(BaseModel):
    """A single turn in the conversation."""
    candidate: str = Field(..., description="Candidate's message")
    recruiter: str = Field(..., description="Recruiter's response")


class GenerateQuestionRequest(BaseModel):
    """Request to generate next interview question."""
    session_id: str = Field(..., description="Unique interview session ID")
    candidate_cv: str = Field(..., description="Candidate's CV/resume text")
    job_role: str = Field(default="AI Engineering Intern", description="Job role title")
    conversation_history: List[ConversationTurn] = Field(
        default=[], description="Previous conversation turns"
    )
    candidate_latest_answer: str = Field(
        ..., description="Candidate's most recent answer"
    )
    max_tokens: int = Field(default=48, ge=1, le=256, description="Max tokens to generate")
    temperature: float = Field(default=0.6, ge=0.0, le=2.0, description="Sampling temperature")


class GenerateQuestionResponse(BaseModel):
    """Response with generated question and metadata."""
    question: str = Field(..., description="Generated interview question")
    reasoning: str = Field(..., description="Why this question was asked")
    topic_depth: int = Field(..., description="Current depth on this topic")
    turn_count: int = Field(..., description="Total interview turns")
    is_follow_up: bool = Field(..., description="Whether this is a follow-up question")
    metrics: Dict[str, Any] = Field(..., description="Generation performance metrics")


class AnalyzeResponseRequest(BaseModel):
    """Request to analyze candidate response."""
    session_id: str = Field(..., description="Interview session ID")
    candidate_answer: str = Field(..., description="Candidate's answer to analyze")


class AnalyzeResponseResponse(BaseModel):
    """Response with answer analysis."""
    word_count: int
    sentence_count: int
    specificity_score: int
    has_specifics: bool
    has_numbers: bool
    is_detailed: bool
    wants_to_skip: bool
    is_nonsense: bool


class SessionStatsResponse(BaseModel):
    """Session statistics."""
    turn_count: int
    topic_depth: int
    consecutive_vague: int
    questions_asked_count: int
    topics_covered: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    warmed_up: bool



@app.on_event("startup")
async def startup_event():
    """Initialize the NLP agent on server startup."""
    global nlp_agent
    logger.info("🚀 Starting NLP Agent server...")

    try:
        nlp_agent = NLPAgent(
            use_compact_prompt=False,  # Use full randomized prompts
            max_model_len=int(os.environ.get("NLP_MAX_MODEL_LEN", "8192")),
        )
        logger.info("✅ NLP Agent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize NLP Agent: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown."""
    logger.info("🛑 Shutting down NLP Agent server...")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns server status and model readiness.
    """
    return HealthResponse(
        status="healthy" if nlp_agent is not None else "initializing",
        model_loaded=nlp_agent is not None,
        warmed_up=nlp_agent._warmup_done if nlp_agent else False,
    )


@app.post("/warmup")
async def warmup(
    job_role: str = "AI Engineering Intern",
    background_tasks: BackgroundTasks = None,
):
    """
    Warm up the model by priming the KV cache.

    This reduces latency for the first real request.
    """
    if nlp_agent is None:
        raise HTTPException(status_code=503, detail="NLP Agent not initialized")

    try:
        await nlp_agent.warmup(job_role=job_role)
        return {"status": "warmup_complete", "job_role": job_role}
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")


@app.post("/generate-question", response_model=GenerateQuestionResponse)
async def generate_question(request: GenerateQuestionRequest):
    """
    Generate the next interview question.

    This is the main endpoint for driving the interview conversation.

    **Logic:**
    - Analyzes the candidate's latest answer
    - Decides whether to ask a follow-up or move to a new topic
    - Generates a contextual question using the fine-tuned model
    - Prevents repetition and handles edge cases

    **Returns:**
    - Generated question
    - Reasoning (why this question)
    - Metadata (topic depth, turn count, metrics)
    """
    if nlp_agent is None:
        raise HTTPException(status_code=503, detail="NLP Agent not initialized")

    try:
        # Convert conversation turns to tuples
        history = [
            (turn.candidate, turn.recruiter)
            for turn in request.conversation_history
        ]

        # Build internal request
        gen_request = QuestionGenerationRequest(
            candidate_cv=request.candidate_cv,
            job_role=request.job_role,
            conversation_history=history,
            candidate_latest_answer=request.candidate_latest_answer,
            session_id=request.session_id,
        )

        # Generate question
        response = await nlp_agent.generate_question(
            request=gen_request,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return GenerateQuestionResponse(
            question=response.question,
            reasoning=response.reasoning,
            topic_depth=response.topic_depth,
            turn_count=response.turn_count,
            is_follow_up=response.is_follow_up,
            metrics={
                "latency_sec": response.metrics.latency_sec,
                "ttft_sec": response.metrics.ttft_sec,
                "num_tokens": response.metrics.num_tokens,
                "tokens_per_sec": response.metrics.tokens_per_sec,
            },
        )

    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Question generation failed: {str(e)}"
        )


@app.post("/analyze-response", response_model=AnalyzeResponseResponse)
async def analyze_response(request: AnalyzeResponseRequest):
    """
    Analyze a candidate's response for depth and quality.

    **Metrics returned:**
    - word_count: Number of words
    - specificity_score: Number of technical specificity keywords found
    - has_specifics: Whether answer contains technical details
    - is_detailed: Overall assessment of answer depth
    - wants_to_skip: Whether candidate wants to skip this topic
    - is_nonsense: Whether answer is non-serious

    **Use case:**
    Provides signals to the scoring agent about communication quality.
    """
    if nlp_agent is None:
        raise HTTPException(status_code=503, detail="NLP Agent not initialized")

    try:
        analysis = nlp_agent.analyze_response(
            session_id=request.session_id,
            candidate_answer=request.candidate_answer,
        )

        return AnalyzeResponseResponse(**analysis)

    except Exception as e:
        logger.error(f"Response analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Response analysis failed: {str(e)}"
        )


@app.get("/session/{session_id}/stats", response_model=SessionStatsResponse)
async def get_session_stats(session_id: str):
    """
    Get statistics about an interview session.

    **Returns:**
    - turn_count: Number of conversation turns
    - topic_depth: Current depth on the current topic
    - consecutive_vague: Number of consecutive vague answers
    - questions_asked_count: Total unique questions asked
    - topics_covered: List of topics covered so far
    """
    if nlp_agent is None:
        raise HTTPException(status_code=503, detail="NLP Agent not initialized")

    try:
        stats = nlp_agent.get_session_stats(session_id)
        return SessionStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Get session stats failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Get session stats failed: {str(e)}"
        )


@app.post("/session/{session_id}/reset")
async def reset_session(session_id: str):
    """
    Reset an interview session.

    Clears all state and starts fresh. Use when restarting an interview.
    """
    if nlp_agent is None:
        raise HTTPException(status_code=503, detail="NLP Agent not initialized")

    try:
        nlp_agent.reset_session(session_id)
        return {"status": "reset", "session_id": session_id}

    except Exception as e:
        logger.error(f"Reset session failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reset session failed: {str(e)}"
        )


@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """
    End an interview session and clean up resources.

    Call this when the interview is complete.
    """
    if nlp_agent is None:
        raise HTTPException(status_code=503, detail="NLP Agent not initialized")

    try:
        nlp_agent.end_session(session_id)
        return {"status": "ended", "session_id": session_id}

    except Exception as e:
        logger.error(f"End session failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"End session failed: {str(e)}"
        )


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "NLP Agent - AI Recruiter",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "generate_question": "POST /generate-question",
            "analyze_response": "POST /analyze-response",
            "session_stats": "GET /session/{id}/stats",
            "reset_session": "POST /session/{id}/reset",
            "end_session": "DELETE /session/{id}",
            "health": "GET /health",
            "warmup": "POST /warmup",
            "docs": "GET /docs (Swagger UI)",
        }
    }
