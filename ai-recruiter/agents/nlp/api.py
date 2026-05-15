"""HTTP router for the NLP agent. Mounted at /api/nlp by the unified app.

This is the live entry point. The legacy standalone server has been
quarantined to apps/_legacy/nlp_demo/ and is no longer wired into the
runtime — keep the old code around for reference only.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# ------------------------------------------------------------------- pydantic
# These models are defined at module scope (not inside build_router) so that
# FastAPI's OpenAPI introspection can resolve their forward references.
# Pydantic 2.12 cannot fully resolve types that live only in a function-local
# closure, which made /docs return 500.
class SessionCreateRequest(BaseModel):
    cv_text: str
    job_role: str = "AI Engineering Intern"


class SessionResponse(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    cv_text: str
    job_role: str
    answer: str
    history: list = []


class TTSRequest(BaseModel):
    text: str


def build_router(agent) -> APIRouter:
    """Build the /api/nlp router bound to an NLPAgentAdapter."""
    router = APIRouter(tags=["nlp"])

    # --------------------------------------------------------------- routes
    @router.get("/health")
    async def health():
        return (await agent.health()).model_dump()

    @router.post("/warmup")
    async def warmup(job_role: str = "AI Engineering Intern",
                     background_tasks: BackgroundTasks = None):
        try:
            await agent.inner.warmup(job_role=job_role)
            return {"status": "warmup_complete", "job_role": job_role}
        except Exception as e:
            raise HTTPException(500, f"Warmup failed: {e}")

    @router.post("/session", response_model=SessionResponse)
    async def create_session(req: SessionCreateRequest):
        sid = f"sess-{uuid.uuid4().hex[:12]}"
        agent.inner._get_or_create_session(sid, req.cv_text)
        agent.inner._get_or_create_prompt(sid, req.cv_text, req.job_role)
        return SessionResponse(session_id=sid)

    @router.delete("/session/{session_id}")
    async def reset_session(session_id: str):
        agent.inner.reset_session(session_id)
        return {"ok": True}

    @router.get("/status")
    async def status():
        return agent.inner.get_status() if agent._inner else {"ready": False}

    @router.get("/stream")
    async def stream_response(
        session_id: str = Query(...),
        cv_text: str = Query(...),
        job_role: str = Query("AI Engineering Intern"),
        answer: str = Query(...),
        history: str = Query("[]"),
    ):
        from .agent import QuestionGenerationRequest
        try:
            history_parsed = json.loads(history)
        except Exception:
            history_parsed = []

        request = QuestionGenerationRequest(
            candidate_cv=cv_text,
            job_role=job_role,
            conversation_history=history_parsed[-6:],
            candidate_latest_answer=answer,
            session_id=session_id,
        )

        async def event_generator():
            try:
                async for chunk in agent.inner.generate_question_streaming_with_audio(request):
                    if chunk.text_delta:
                        yield f"data: {json.dumps({'type':'token','text':chunk.text_delta})}\n\n"
                    if chunk.tts_result is not None:
                        b64 = base64.b64encode(chunk.tts_result.to_wav_bytes()).decode("utf-8")
                        yield f"data: {json.dumps({'type':'audio','data':b64})}\n\n"
                    if chunk.is_final:
                        yield f"data: {json.dumps({'type':'done'})}\n\n"
                        return
            except Exception as e:
                yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/chat")
    async def chat_non_streaming(req: ChatRequest):
        from .agent import QuestionGenerationRequest
        request = QuestionGenerationRequest(
            candidate_cv=req.cv_text,
            job_role=req.job_role,
            conversation_history=req.history[-6:],
            candidate_latest_answer=req.answer,
            session_id=req.session_id,
        )
        response = await agent.inner.generate_question(request)
        audio_b64 = None
        if response.audio:
            try:
                audio_b64 = base64.b64encode(response.audio.to_wav_bytes()).decode("utf-8")
            except Exception:
                pass
        return {
            "question": response.question,
            "reasoning": response.reasoning,
            "is_follow_up": response.is_follow_up,
            "audio_b64": audio_b64,
            "metrics": {
                "latency_sec": response.metrics.latency_sec if response.metrics else None,
                "num_tokens": response.metrics.num_tokens if response.metrics else None,
            },
        }

    @router.post("/tts")
    async def text_to_speech(req: TTSRequest):
        if not agent.inner._tts or not agent.inner._tts.is_ready:
            raise HTTPException(503, "TTS not enabled")
        try:
            audio_result = await agent.inner._tts.synthesize(req.text)
            audio_b64 = base64.b64encode(audio_result.to_wav_bytes()).decode("utf-8")
            return {"audio_b64": audio_b64}
        except Exception as e:
            raise HTTPException(500, f"TTS generation failed: {e}")

    return router
