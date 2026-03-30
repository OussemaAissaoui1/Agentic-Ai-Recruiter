"""
Streamlit Demo — AI Recruiter Interview Simulator

Usage:
    cd agents/nlp
    streamlit run demo/app.py

Features:
    - Chat-like interface
    - Audio playback of recruiter responses
    - Real-time metrics display
    - Scorer analysis visibility
    - Session management
    - Production-ready error handling
"""

import sys
import os

# Add parent directory to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import asyncio
import concurrent.futures
import threading
import queue
import time
import uuid
from typing import Optional

# Page config must be first Streamlit command
st.set_page_config(
    page_title="AI Recruiter — Interview Simulator",
    page_icon="🎤",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Async helper — persistent background loop (fixes vLLM event-loop mismatch)
# ---------------------------------------------------------------------------
#
# ROOT CAUSE OF THE "STUCK AT THINKING" BUG:
#   Streamlit's main thread has a running event loop (tornado's).
#   The old run_async called asyncio.run(coro) inside a ThreadPoolExecutor,
#   which created a BRAND NEW event loop for every request.
#   vLLM's AsyncLLMEngine was initialised (during warmup) on one of those
#   short-lived loops; when the next request ran in yet another new loop,
#   vLLM's internal queue never got a response → infinite hang.
#
# FIX:
#   One persistent background thread owns a single event loop that lives
#   for the entire process lifetime.  ALL async work — warmup, generate,
#   TTS, scorer — is scheduled onto that loop via run_coroutine_threadsafe.
#   vLLM is always called from the same loop it was initialised in.

_BG_LOOP: Optional[asyncio.AbstractEventLoop] = None
_BG_THREAD: Optional[threading.Thread] = None
_BG_LOOP_LOCK = threading.Lock()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    """Return the single persistent background event loop, creating it if needed."""
    global _BG_LOOP, _BG_THREAD
    if _BG_LOOP is not None and _BG_LOOP.is_running():
        return _BG_LOOP

    with _BG_LOOP_LOCK:
        # Double-checked locking
        if _BG_LOOP is not None and _BG_LOOP.is_running():
            return _BG_LOOP

        loop = asyncio.new_event_loop()

        def _run(lp: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(lp)
            lp.run_forever()

        t = threading.Thread(target=_run, args=(loop,), daemon=True, name="StreamlitAsyncBg")
        t.start()

        # Spin-wait until the loop is actually running (usually <5 ms)
        deadline = time.monotonic() + 2.0
        while not loop.is_running():
            if time.monotonic() > deadline:
                raise RuntimeError("Background event loop failed to start")
            time.sleep(0.001)

        _BG_LOOP = loop
        _BG_THREAD = t
        return _BG_LOOP


def run_async(coro, timeout: int = 120):
    """
    Run an async coroutine from synchronous Streamlit code.

    Always schedules on the persistent background loop so that vLLM (and
    every other async component) sees the same event loop across all calls.
    """
    loop = _get_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise TimeoutError(f"Async operation timed out after {timeout}s")



def run_async_streaming(gen_coro, placeholder, timeout: int = 30) -> str:
    """
    Stream tokens from an async generator while updating Streamlit safely.

    The async generator runs on the persistent background loop and pushes
    chunks into a thread-safe queue. All Streamlit UI calls happen here,
    on the main thread where ScriptRunContext exists — NOT in the background
    thread (which causes 'missing ScriptRunContext' hangs and timeouts).

    Args:
        gen_coro: the async generator (agent.generate_question_streaming(...))
        placeholder: a st.empty() placeholder for incremental display
        timeout: seconds to wait for the next chunk before giving up

    Returns:
        The full concatenated response string.
    """
    loop = _get_background_loop()
    chunk_queue: "queue.Queue[object]" = queue.Queue()
    _SENTINEL = object()

    async def _collect() -> None:
        try:
            async for chunk in gen_coro:
                chunk_queue.put(chunk)
        finally:
            chunk_queue.put(_SENTINEL)

    future = asyncio.run_coroutine_threadsafe(_collect(), loop)

    chunks = []
    while True:
        try:
            item = chunk_queue.get(timeout=timeout)
        except queue.Empty:
            future.cancel()
            raise TimeoutError(f"Streaming timed out after {timeout}s with no new tokens")

        if item is _SENTINEL:
            break

        chunks.append(item)
        # Safe: this is the main Streamlit thread
        placeholder.markdown("".join(chunks) + "\u25cc")

    # Propagate any exception from the background coroutine
    try:
        future.result(timeout=5)
    except Exception as exc:
        raise RuntimeError(f"Streaming coroutine failed: {exc}") from exc

    full = "".join(chunks)
    placeholder.markdown(full)
    return full

# ---------------------------------------------------------------------------
# GPU Health Check
# ---------------------------------------------------------------------------

def check_gpu_health():
    """Check GPU status before loading models."""
    try:
        from vllm_engine import check_gpu_status, cleanup_gpu_processes
        status = check_gpu_status()
        return status
    except Exception as e:
        return None


def display_gpu_status(location: str = "sidebar"):
    """Display GPU status in sidebar."""
    status = check_gpu_health()
    if status is None:
        st.sidebar.error("Unable to check GPU status")
        return False

    if not status.available:
        st.sidebar.error(f"GPU unavailable: {status.error}")
        return False

    st.sidebar.success(
        f"GPU: {status.free_memory_gb:.1f}GB free / {status.total_memory_gb:.1f}GB total"
    )

    if status.blocking_pids:
        st.sidebar.warning(f"Blocking processes: {status.blocking_pids}")
        if st.sidebar.button("🧹 Cleanup GPU Processes", key=f"cleanup_btn_{location}"):
            from vllm_engine import cleanup_gpu_processes
            killed = cleanup_gpu_processes(force=False)
            st.sidebar.info(f"Killed {killed} processes")
            st.rerun()

    return True


# ---------------------------------------------------------------------------
# Agent initialization (cached so model loads only once)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_agent():
    """Load the NLP agent with all models. Cached across reruns."""
    from agent import NLPAgent

    agent = NLPAgent(
        enable_scorer=True, # Enable CPU scorer for intelligent answer analysis
        scorer_model="Qwen/Qwen2.5-1.5B-Instruct",
        enable_tts=True,  # Enable TTS for voice output
        tts_voice="af_heart",
        use_compact_prompt=True,
        auto_cleanup_gpu=False,  # Don't auto-cleanup to avoid killing ourselves
        load_timeout_sec=180,
    )   
    return agent


def initialize_agent_with_progress():
    """Initialize agent with progress display."""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    try:
        # Step 1: Check GPU
        with progress_placeholder.container():
            st.progress(0.1, "Checking GPU availability...")

        gpu_ok = display_gpu_status("init")
        if not gpu_ok:
            st.error("GPU not available. Please check your GPU setup.")
            st.stop()

        # Step 2: Load agent (cached)
        with progress_placeholder.container():
            st.progress(0.3, "Loading NLP Agent...")

        agent = load_agent()

        # Step 3: Warmup if not done
        if not agent._warmup_done:
            with progress_placeholder.container():
                st.progress(0.5, "Loading vLLM model (this takes ~60 seconds)...")

            try:
                run_async(agent.warmup(), timeout=180)
            except Exception as e:
                st.error(f"Model warmup failed: {e}")
                st.info("Try refreshing the page. If the problem persists, check GPU memory.")
                st.stop()

        # Step 4: Done
        with progress_placeholder.container():
            st.progress(1.0, "Ready!")

        time.sleep(0.5)
        progress_placeholder.empty()
        status_placeholder.empty()

        return agent

    except Exception as e:
        progress_placeholder.empty()
        st.error(f"Failed to initialize agent: {e}")
        st.info("Tips: 1) Check GPU memory 2) Try 'Cleanup GPU Processes' 3) Restart the app")
        st.stop()


# ---------------------------------------------------------------------------
# Default CV for demo
# ---------------------------------------------------------------------------

DEFAULT_CV = """Candidate: Oussema Aissaoui
Role applied for: AI Engineering Intern

Education:
- National School of Computer Science (ENSI), Computer Engineering, Sep 2023–Present
  Coursework: AI, Data Analysis, Machine Learning, Software Engineering

Experience:
- Technology Intern, TALAN Tunisia, Jul–Aug 2025
  Applied GraphRAG and LLM reasoning for pattern recognition in enterprise datasets.
  Designed AI-driven prototypes for data analytics and business decision systems.
- AI & Automation Intern, NP Tunisia, Jul–Aug 2024
  Automated integration of 30+ industrial screens into a local network.
  Standardized config parameters and automated monitoring to eliminate deployment errors.

Projects:
- Pattern Recognition in Company Internal Data: GraphRAG + LLM reasoning on enterprise datasets.
- Blood Cell Segmentation: YOLOv8 + SAM pipeline for WBC/RBC classification and anomaly detection.
- Water Use Efficiency Estimation: ML model using WaPOR + Google Earth Engine for crop evapotranspiration.
- Blockchain E-Commerce: Decentralized marketplace using Solidity smart contracts.

Technical Skills:
- Languages: Python, C/C++, Java, JavaScript, Solidity
- AI & Data: ML, Deep Learning, TensorFlow, OpenCV, YOLOv8, SAM, NLP, GraphRAG
"""


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"demo-{uuid.uuid4().hex[:8]}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "started" not in st.session_state:
        st.session_state.started = False
    if "cv_text" not in st.session_state:
        st.session_state.cv_text = DEFAULT_CV
    if "job_role" not in st.session_state:
        st.session_state.job_role = "AI Engineering Intern"


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main():
    init_session()

    # Main area header
    st.title("🎤 AI Recruiter — Interview Simulator")
    st.caption("Type your answers as the candidate. The AI recruiter will respond with voice.")

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")

        st.session_state.job_role = st.text_input(
            "Job Role", value=st.session_state.job_role
        )

        st.session_state.cv_text = st.text_area(
            "Candidate CV",
            value=st.session_state.cv_text,
            height=300,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", use_container_width=True):
                if "agent_loaded" in st.session_state:
                    agent = load_agent()
                    agent.reset_session(st.session_state.session_id)
                st.session_state.messages = []
                st.session_state.history = []
                st.session_state.started = False
                st.rerun()

        with col2:
            show_debug = st.checkbox("Debug", value=False)

        use_streaming = st.checkbox(
            "Fast streaming mode",
            value=False,
            help="Lower latency but less robust metadata/scoring. Disable for most stable vocal demo.",
        )

        st.divider()

        # GPU status
        display_gpu_status("main")

        st.divider()
        st.caption("Models:")

        # Show model status (only if agent is loaded)
        if "agent_loaded" in st.session_state and st.session_state.agent_loaded:
            agent = load_agent()
            status = agent.get_status()
            if status["engine_ready"]:
                st.caption("• Recruiter: LLaMA 3.1 8B (GPU) ✅")
            else:
                st.caption(f"• Recruiter: ❌ {status.get('engine_error', 'Not ready')}")

            if agent._scorer and agent._scorer.is_ready:
                st.caption("• Scorer: Qwen 1.5B (CPU) ✅")
            else:
                st.caption("• Scorer: Disabled ❌")

            if agent._tts and agent._tts.is_ready:
                st.caption("• TTS: Kokoro (CPU) ✅")
            else:
                st.caption("• TTS: Disabled ❌")
        else:
            st.caption("• Models not loaded yet")

    # Initialize agent with progress (only on first load)
    if "agent_loaded" not in st.session_state:
        agent = initialize_agent_with_progress()
        st.session_state.agent_loaded = True
        st.rerun()

    agent = load_agent()

    # Display chat history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar=msg.get("avatar")):
            st.write(msg["content"])

            # Show audio player if available
            if msg.get("audio_bytes"):
                # Autoplay only if it's the newest message and flag is set
                do_autoplay = msg.get("autoplay", False)
                st.audio(msg["audio_bytes"], format="audio/wav", autoplay=do_autoplay)
                if do_autoplay:
                    msg["autoplay"] = False  # play only once

            # Show metrics if debug
            if show_debug and msg.get("metrics"):
                m = msg["metrics"]
                cols = st.columns(4)
                cols[0].metric("TTFT", f"{m.get('ttft', 'N/A')}")
                cols[1].metric("Latency", f"{m.get('latency', 'N/A')}")
                cols[2].metric("Tokens", f"{m.get('tokens', 'N/A')}")
                cols[3].metric("Reason", f"{m.get('reason', 'N/A')}")

            if show_debug and msg.get("scorer"):
                with st.expander("Scorer Analysis"):
                    st.json(msg["scorer"])

    # Chat input
    if prompt := st.chat_input("Type your answer..."):
        agent = load_agent()

        # Show user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "avatar": "👤",
        })
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)

        # Generate response with streaming
        with st.chat_message("assistant", avatar="🎤"):
            message_placeholder = st.empty()
            full_response = ""
            response_obj = None

            try:
                from agent import QuestionGenerationRequest

                request = QuestionGenerationRequest(
                    candidate_cv=st.session_state.cv_text,
                    job_role=st.session_state.job_role,
                    conversation_history=st.session_state.history[-6:],
                    candidate_latest_answer=prompt,
                    session_id=st.session_state.session_id,
                )

                if use_streaming:
                    # Fast streaming path — uses run_async_streaming so that
                    # all Streamlit UI calls stay on the main thread (fixes the
                    # "missing ScriptRunContext" timeout caused by calling
                    # message_placeholder.markdown from the background loop).
                    t_start = time.time()

                    full_response = run_async_streaming(
                        agent.generate_question_streaming(request),
                        message_placeholder,
                        timeout=30,
                    )
                    full_response = " ".join(full_response.split())
                    message_placeholder.markdown(full_response)

                    latency = time.time() - t_start
                    metrics = type("metrics", (object,), {
                        "latency_sec": latency,
                        "ttft_sec": None,
                        "num_tokens": len(full_response.split()),
                        "tokens_per_sec": len(full_response.split()) / latency if latency > 0 else 0,
                    })()
                    response_obj = type("obj", (object,), {
                        "question": full_response,
                        "reasoning": "streaming_fast_path",
                        "topic_depth": 0,
                        "turn_count": 0,
                        "metrics": metrics,
                        "is_follow_up": False,
                        "scorer_output": None,
                        "audio": None,
                    })()

                    # Ensure vocal output works in streaming mode too.
                    if getattr(agent, "_tts", None) and agent._tts.is_ready and full_response:
                        try:
                            response_obj.audio = run_async(agent._tts.synthesize(full_response), timeout=10)
                        except Exception as tts_err:
                            print(f"[app] Streaming TTS error: {tts_err}")
                else:
                    # Stable production path: full generation + validation + scorer + TTS.
                    message_placeholder.markdown("🤔 Thinking...")
                    response_obj = run_async(agent.generate_question(request), timeout=45)
                    full_response = response_obj.question
                    message_placeholder.markdown(full_response)

            except Exception as e:
                message_placeholder.error(f"Error: {e}")
                print(f"[app] Generation error: {e}")
                import traceback
                traceback.print_exc()
                st.stop()

            # Convert audio for Streamlit playback
            audio_bytes = None
            if hasattr(response_obj, 'audio') and response_obj.audio:
                try:
                    audio_bytes = response_obj.audio.to_wav_bytes()
                except Exception as e:
                    print(f"[app] Audio conversion error: {e}")

            # Metrics
            if hasattr(response_obj, 'metrics') and response_obj.metrics:
                ttft_str = f"{response_obj.metrics.ttft_sec:.2f}s" if response_obj.metrics.ttft_sec else "N/A"
                indicator = "🔄" if response_obj.is_follow_up else "➡️"

                st.caption(
                    f"{indicator} {response_obj.reasoning} | "
                    f"TTFT: {ttft_str} | "
                    f"Latency: {response_obj.metrics.latency_sec:.2f}s | "
                    f"Tokens: {response_obj.metrics.num_tokens}"
                )

                if show_debug and response_obj.scorer_output:
                    with st.expander("Scorer Analysis"):
                        st.json(response_obj.scorer_output)

        # Update state
        st.session_state.history.append((prompt, full_response))
        metrics_dict = {
            "ttft": ttft_str if hasattr(response_obj, 'metrics') and response_obj.metrics else "N/A",
            "latency": f"{response_obj.metrics.latency_sec:.2f}s" if hasattr(response_obj, 'metrics') and response_obj.metrics else "N/A",
            "tokens": response_obj.metrics.num_tokens if hasattr(response_obj, 'metrics') and response_obj.metrics else 0,
            "reason": response_obj.reasoning if hasattr(response_obj, 'reasoning') else "unknown",
        }
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "avatar": "🎤",
            "audio_bytes": audio_bytes,
            "autoplay": True,
            "metrics": metrics_dict,
            "scorer": response_obj.scorer_output if hasattr(response_obj, 'scorer_output') else None,
        })

        st.rerun()


if __name__ == "__main__":
    main()