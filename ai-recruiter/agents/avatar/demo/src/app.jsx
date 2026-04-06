/**
 * AI Recruiter Avatar Demo — Main Application
 *
 * Full interview interface with:
 *   - 3D GLB avatar with real-time lip sync (replaces pulsing orb)
 *   - Streaming SSE with text tokens + audio + server-side visemes
 *   - Client-side frequency analysis for smooth lip transitions
 *   - Chat interface with message history
 *   - Session management with CV/role setup
 *
 * Architecture:
 *   Browser ◄─SSE─► FastAPI (NLP agent + Avatar agent)
 *   Audio ──► LipSyncAudioQueue ──► AnalyserNode ──► Viseme weights
 *   Three.js ◄── Viseme weights ──► GLB morph targets
 */

import { useState, useEffect, useRef, useCallback } from "react";
import AvatarCanvas from "./AvatarCanvas.jsx";
import { LipSyncAudioQueue } from "./AudioLipSync.js";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const API_BASE = "";
const DEFAULT_CV = `Candidate: Oussema Aissaoui
Role applied for: AI Engineering Intern
Education:
- National School of Computer Science (ENSI), Computer Engineering, Sep 2023–Present
Experience:
- Technology Intern, TALAN Tunisia, Jul–Aug 2025
  Applied GraphRAG and LLM reasoning for pattern recognition in enterprise datasets.
- AI & Automation Intern, NP Tunisia, Jul–Aug 2024
  Automated integration of 30+ industrial screens into a local network.
Projects:
- Pattern Recognition in Company Internal Data: GraphRAG + LLM reasoning
- Blood Cell Segmentation: YOLOv8 + SAM pipeline for WBC/RBC classification
- Water Use Efficiency Estimation: ML model using WaPOR + Google Earth Engine
- Blockchain E-Commerce: Decentralized marketplace using Solidity smart contracts
Technical Skills:
- Languages: Python, C/C++, Java, JavaScript, Solidity
- AI & Data: ML, Deep Learning, TensorFlow, OpenCV, YOLOv8, SAM, NLP, GraphRAG`;

// ---------------------------------------------------------------------------
// Error Boundary — catches Three.js / Canvas rendering errors
// ---------------------------------------------------------------------------
import React from "react";

class AvatarErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error("[AvatarErrorBoundary]", error, info?.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <ErrorFallback
          error={this.state.error}
          resetFn={() => this.setState({ hasError: false, error: null })}
        />
      );
    }
    return this.props.children;
  }
}

function ErrorFallback({ error, resetFn }) {
  return (
    <div style={{
      padding: 24, textAlign: "center", color: "#fca5a5",
      background: "rgba(239,68,68,0.1)", borderRadius: 12, margin: 16,
    }}>
      <h3 style={{ margin: "0 0 8px" }}>3D Avatar Error</h3>
      <p style={{ fontSize: 13, color: "#94a3b8", margin: "0 0 12px" }}>
        {error?.message || "Failed to load 3D avatar. Falling back to text-only mode."}
      </p>
      {resetFn && (
        <button onClick={resetFn} style={{
          padding: "6px 16px", borderRadius: 6,
          background: "rgba(99,102,241,0.3)", border: "1px solid rgba(99,102,241,0.5)",
          color: "#818cf8", cursor: "pointer", fontSize: 13,
        }}>
          Retry
        </button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Message bubble
// ---------------------------------------------------------------------------
function Message({ role, text, isStreaming }) {
  const isRecruiter = role === "recruiter";
  return (
    <div style={{
      display: "flex",
      justifyContent: isRecruiter ? "flex-start" : "flex-end",
      marginBottom: 12,
    }}>
      <div style={{
        maxWidth: "72%",
        padding: "12px 16px",
        borderRadius: isRecruiter ? "4px 18px 18px 18px" : "18px 4px 18px 18px",
        background: isRecruiter ? "rgba(99,102,241,0.12)" : "rgba(255,255,255,0.07)",
        border: `1px solid ${isRecruiter ? "rgba(99,102,241,0.3)" : "rgba(255,255,255,0.1)"}`,
        color: "#e2e8f0",
        fontSize: 15,
        lineHeight: 1.6,
        letterSpacing: 0.1,
      }}>
        {text}
        {isStreaming && (
          <span style={{
            display: "inline-block", width: 8, height: 14,
            background: "#818cf8", marginLeft: 3, verticalAlign: "middle",
            borderRadius: 1, animation: "blink 0.8s step-end infinite",
          }} />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Setup Screen
// ---------------------------------------------------------------------------
function SetupScreen({ onStart }) {
  const [cv, setCv] = useState(DEFAULT_CV);
  const [role, setRole] = useState("AI Engineering Intern");
  const [candidateName, setCandidateName] = useState("Oussema Aissaoui");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState(null);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();
        setStatus(data);
        setStatusError(null);
      } catch (e) {
        console.error("[Setup] Status check failed:", e);
        setStatus({ ready: false });
        setStatusError("Cannot reach server");
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleStart = async () => {
    if (!cv.trim() || !role.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cv_text: cv, job_role: role }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Session creation failed");
      }
      const data = await res.json();
      console.log("[Setup] Session created:", data.session_id);
      onStart({ sessionId: data.session_id, cv, role, candidateName });
    } catch (e) {
      console.error("[Setup] Failed:", e);
      alert("Failed to create session: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  const inputStyle = {
    width: "100%",
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(255,255,255,0.12)",
    borderRadius: 10,
    color: "#e2e8f0",
    padding: "10px 14px",
    fontSize: 14,
    outline: "none",
    boxSizing: "border-box",
    resize: "vertical",
  };

  const isReady = status?.ready;
  const avatarOk = status?.avatar_ready;
  const nlpOk = status?.nlp_ready;

  return (
    <div style={{
      minHeight: "100vh", background: "#0f0f1a",
      display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
    }}>
      <div style={{ width: "100%", maxWidth: 560 }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <div style={{
            width: 100, height: 100, borderRadius: "50%",
            background: "linear-gradient(135deg, #6366f1, #4f46e5)",
            margin: "0 auto 16px", display: "flex",
            alignItems: "center", justifyContent: "center", fontSize: 42,
          }}>
            👤
          </div>
          <h1 style={{ color: "#e2e8f0", fontSize: 26, margin: "0 0 6px", fontWeight: 700 }}>
            AI Recruiter
          </h1>
          <p style={{ color: "#94a3b8", fontSize: 14, margin: 0 }}>
            3D Avatar Interview Simulator
          </p>
        </div>

        {/* Status badges */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
          {[
            { label: "NLP Agent", ok: nlpOk },
            { label: "Avatar", ok: avatarOk },
            { label: "GPU", ok: status?.engine_ready, extra: status?.gpu_free_gb ? `${status.gpu_free_gb}GB` : null },
          ].map(({ label, ok, extra }) => (
            <div key={label} style={{
              flex: 1, padding: "8px 12px",
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 8, fontSize: 12,
              color: ok ? "#86efac" : ok === false ? "#fca5a5" : "#94a3b8",
              display: "flex", alignItems: "center", gap: 6,
            }}>
              <div style={{
                width: 6, height: 6, borderRadius: "50%",
                background: ok ? "#22c55e" : ok === false ? "#ef4444" : "#94a3b8",
                flexShrink: 0,
              }} />
              {label}{extra ? ` · ${extra}` : ""}
            </div>
          ))}
        </div>

        {statusError && (
          <div style={{
            padding: "8px 14px", marginBottom: 16, borderRadius: 8,
            background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.2)",
            color: "#fca5a5", fontSize: 13,
          }}>
            {statusError}
          </div>
        )}

        {/* Form */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <label style={{ color: "#94a3b8", fontSize: 13, display: "block", marginBottom: 6 }}>
              Candidate Name
            </label>
            <input
              value={candidateName}
              onChange={(e) => setCandidateName(e.target.value)}
              placeholder="Enter candidate's full name"
              style={{ ...inputStyle, resize: "none" }}
            />
          </div>
          <div>
            <label style={{ color: "#94a3b8", fontSize: 13, display: "block", marginBottom: 6 }}>
              Job Role
            </label>
            <input
              value={role}
              onChange={(e) => setRole(e.target.value)}
              style={{ ...inputStyle, resize: "none" }}
            />
          </div>
          <div>
            <label style={{ color: "#94a3b8", fontSize: 13, display: "block", marginBottom: 6 }}>
              Candidate CV
            </label>
            <textarea
              value={cv}
              onChange={(e) => setCv(e.target.value)}
              rows={10}
              style={inputStyle}
            />
          </div>
          <button
            onClick={handleStart}
            disabled={loading || !isReady}
            style={{
              padding: "13px 0", borderRadius: 10,
              background: loading || !isReady
                ? "rgba(99,102,241,0.3)"
                : "linear-gradient(135deg, #6366f1, #4f46e5)",
              border: "none", color: "#fff", fontSize: 15, fontWeight: 600,
              cursor: loading || !isReady ? "not-allowed" : "pointer",
              letterSpacing: 0.3, transition: "opacity 0.2s",
            }}
          >
            {loading ? "Starting…" : "Start Interview"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Interview Screen — 3D Avatar + Chat
// ---------------------------------------------------------------------------
function InterviewScreen({ sessionId, cv, role, candidateName, onReset }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [history, setHistory] = useState([]);
  const [avatarError, setAvatarError] = useState(null);

  const lipSyncRef = useRef(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  // Initialize lip-sync audio queue
  useEffect(() => {
    const lsq = new LipSyncAudioQueue();
    lsq.onActivityChange = setIsSpeaking;
    lipSyncRef.current = lsq;

    console.log("[Interview] LipSyncAudioQueue initialized");

    return () => {
      console.log("[Interview] Cleaning up audio queue");
      lsq.destroy();
    };
  }, []);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  // --- SSE stream handler (shared between greeting and messages) ---
  const handleSSEStream = useCallback(async (url, userText, isGreeting = false) => {
    const lipSync = lipSyncRef.current;

    try {
      const response = await fetch(url, {
        signal: abortRef.current?.signal,
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullText = "";
      let pendingVisemes = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          let event;
          try {
            event = JSON.parse(raw);
          } catch {
            console.warn("[SSE] Failed to parse event:", raw.slice(0, 100));
            continue;
          }

          switch (event.type) {
            case "token":
              fullText += event.text;
              setStreamingText(fullText);
              setIsThinking(false);
              break;

            case "audio":
              // Enqueue audio with any pending visemes
              if (lipSync) {
                lipSync.enqueue(event.data, pendingVisemes);
              }
              pendingVisemes = null;
              break;

            case "visemes":
              // Store visemes to attach to the next audio chunk
              pendingVisemes = event.data;
              break;

            case "done": {
              const finalText = fullText.trim();

              if (isGreeting) {
                setMessages([{ role: "recruiter", text: finalText }]);
                setHistory([["hey", finalText]]);
              } else {
                setMessages((prev) => [
                  ...prev,
                  { role: "recruiter", text: finalText },
                ]);
                setHistory((prev) => [...prev, [userText, finalText]]);
              }

              setStreamingText("");
              setIsStreaming(false);
              setIsThinking(false);
              inputRef.current?.focus();

              console.log(
                `[SSE] Stream complete | chars=${finalText.length}`
              );
              return;
            }

            case "error":
              console.error("[SSE] Server error:", event.message);
              setMessages((prev) => [
                ...prev,
                { role: "recruiter", text: "Something went wrong. Please try again." },
              ]);
              setStreamingText("");
              setIsStreaming(false);
              setIsThinking(false);
              return;

            default:
              console.warn("[SSE] Unknown event type:", event.type);
          }
        }
      }

      // Safety: stream ended without "done" event
      if (fullText.trim()) {
        console.warn("[SSE] Stream ended without done event");
        const finalText = fullText.trim();
        setMessages((prev) => {
          if (isGreeting && prev.length === 0) {
            return [{ role: "recruiter", text: finalText }];
          }
          return [...prev, { role: "recruiter", text: finalText }];
        });
        if (isGreeting) {
          setHistory((prev) => (prev.length === 0 ? [["hey", finalText]] : prev));
        } else {
          setHistory((prev) => [...prev, [userText, finalText]]);
        }
      }
      setStreamingText("");
      setIsStreaming(false);
      setIsThinking(false);
    } catch (e) {
      if (e.name === "AbortError") {
        console.log("[SSE] Stream aborted");
      } else {
        console.error("[SSE] Fetch error:", e);
        setMessages((prev) => [
          ...prev,
          { role: "recruiter", text: "Connection error. Please try again." },
        ]);
      }
      setStreamingText("");
      setIsStreaming(false);
      setIsThinking(false);
    }
  }, []);

  // Opening greeting — guarded against React StrictMode double-mount
  const greetingFired = useRef(false);
  useEffect(() => {
    if (greetingFired.current) return;
    greetingFired.current = true;

    const controller = new AbortController();
    abortRef.current = controller;

    const fetchGreeting = async () => {
      setIsThinking(true);
      setIsStreaming(true);

      const params = new URLSearchParams({
        session_id: sessionId,
        cv_text: cv,
        job_role: role,
        answer: "hey",
        history: JSON.stringify([]),
      });

      await handleSSEStream(
        `${API_BASE}/api/stream?${params.toString()}`,
        "hey",
        true
      );
    };

    fetchGreeting();

    return () => {
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Send message
  const sendMessage = useCallback(
    async (userText) => {
      if (isStreaming || !userText.trim()) return;

      console.log("[Chat] Sending:", userText.slice(0, 50));

      setMessages((prev) => [...prev, { role: "user", text: userText }]);
      setInput("");
      setIsThinking(true);
      setIsStreaming(true);
      setStreamingText("");
      lipSyncRef.current?.clear();

      // Abort previous stream
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const params = new URLSearchParams({
        session_id: sessionId,
        cv_text: cv,
        job_role: role,
        answer: userText,
        history: JSON.stringify(history),
      });

      await handleSSEStream(
        `${API_BASE}/api/stream?${params.toString()}`,
        userText,
        false
      );
    },
    [sessionId, cv, role, history, isStreaming, handleSSEStream]
  );

  const handleSubmit = () => {
    if (input.trim() && !isStreaming) {
      sendMessage(input.trim());
    }
  };

  const handleInputFocus = () => {
    // Resume AudioContext on user interaction (browser requirement)
    try {
      const lsq = lipSyncRef.current;
      if (lsq) lsq._getCtx();
    } catch (e) {
      console.warn("[Audio] Context resume failed:", e);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleReset = async () => {
    abortRef.current?.abort();
    lipSyncRef.current?.clear();
    try {
      await fetch(`${API_BASE}/api/session/${sessionId}`, { method: "DELETE" });
    } catch (e) {
      console.warn("[Reset] Session delete failed:", e);
    }
    onReset();
  };

  const orbLabel = isThinking
    ? "Thinking…"
    : isSpeaking
    ? "Speaking…"
    : isStreaming
    ? "Generating…"
    : "Listening";

  return (
    <div style={{
      height: "100vh", background: "#0f0f1a",
      display: "flex", flexDirection: "row", overflow: "hidden",
    }}>
      {/* ═══════════════════════════════════════════════════ */}
      {/* LEFT — Avatar panel (60%)                          */}
      {/* ═══════════════════════════════════════════════════ */}
      <div style={{
        width: "60%", height: "100vh", flexShrink: 0, position: "relative",
        background: "radial-gradient(ellipse at 50% 35%, #1a1a38 0%, #0f0f1a 70%)",
        borderRight: "1px solid rgba(255,255,255,0.06)",
      }}>
        {avatarError ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
            <ErrorFallback error={avatarError} resetFn={() => setAvatarError(null)} />
          </div>
        ) : (
          <AvatarCanvasWrapper
            lipSyncRef={lipSyncRef}
            isSpeaking={isSpeaking}
            isThinking={isThinking}
            onError={setAvatarError}
          />
        )}

        {/* Status badge overlay */}
        <div style={{
          position: "absolute", bottom: 28, left: "50%", transform: "translateX(-50%)",
          padding: "6px 18px", borderRadius: 20,
          background: "rgba(0,0,0,0.55)", backdropFilter: "blur(10px)",
          fontSize: 13, letterSpacing: 0.5, pointerEvents: "none",
          color: isSpeaking ? "#818cf8" : isThinking ? "#a78bfa" : "#475569",
          transition: "color 0.3s",
          border: "1px solid rgba(255,255,255,0.08)",
        }}>
          {isSpeaking ? "● Speaking" : isThinking ? "● Thinking…" : isStreaming ? "● Generating…" : "● Listening"}
        </div>

        {/* Candidate name badge */}
        <div style={{
          position: "absolute", top: 20, left: 20,
          padding: "5px 14px", borderRadius: 20,
          background: "rgba(0,0,0,0.45)", backdropFilter: "blur(8px)",
          fontSize: 12, color: "#94a3b8", pointerEvents: "none",
          border: "1px solid rgba(255,255,255,0.07)", letterSpacing: 0.3,
        }}>
          {candidateName}
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════ */}
      {/* RIGHT — Chat panel (40%)                           */}
      {/* ═══════════════════════════════════════════════════ */}
      <div style={{
        width: "40%", height: "100vh",
        display: "flex", flexDirection: "column",
        background: "#0f0f1a",
      }}>
        {/* ── Top bar ── */}
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "14px 20px",
          borderBottom: "1px solid rgba(255,255,255,0.07)",
          background: "rgba(15,15,26,0.95)", backdropFilter: "blur(12px)",
          flexShrink: 0,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{
              width: 32, height: 32, borderRadius: "50%",
              background: "linear-gradient(135deg, #6366f1, #4f46e5)",
              display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16,
            }}>
              👤
            </div>
            <div>
              <div style={{ color: "#e2e8f0", fontSize: 14, fontWeight: 600 }}>
                Alex · AI Recruiter
              </div>
              <div style={{ color: "#64748b", fontSize: 12 }}>{role}</div>
            </div>
          </div>
          <button
            onClick={handleReset}
            style={{
              padding: "6px 14px", borderRadius: 7,
              background: "rgba(255,255,255,0.05)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: "#94a3b8", fontSize: 13, cursor: "pointer",
            }}
          >
            New Interview
          </button>
        </div>

        {/* ── Messages ── */}
        <div style={{
          flex: 1, overflowY: "auto", padding: "20px 18px",
          boxSizing: "border-box",
        }}>
          {messages.map((msg, i) => (
            <Message key={i} role={msg.role} text={msg.text} isStreaming={false} />
          ))}
          {streamingText && (
            <Message role="recruiter" text={streamingText} isStreaming={true} />
          )}
          {isThinking && !streamingText && (
            <div style={{ display: "flex", gap: 6, padding: "8px 0", paddingLeft: 4 }}>
              {[0, 1, 2].map((i) => (
                <div key={i} style={{
                  width: 8, height: 8, borderRadius: "50%", background: "#6366f1",
                  animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
                }} />
              ))}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* ── Input ── */}
        <div style={{
          padding: "14px 18px",
          borderTop: "1px solid rgba(255,255,255,0.07)",
          background: "rgba(15,15,26,0.95)", backdropFilter: "blur(12px)",
          flexShrink: 0,
        }}>
          <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={handleInputFocus}
              placeholder={isStreaming ? "Wait for Alex to finish…" : "Type your answer…"}
              disabled={isStreaming}
              rows={1}
              style={{
                flex: 1,
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 12, color: "#e2e8f0",
                padding: "12px 14px", fontSize: 14, outline: "none",
                resize: "none", lineHeight: 1.5, maxHeight: 120, overflow: "auto",
                opacity: isStreaming ? 0.5 : 1, transition: "opacity 0.2s",
              }}
              onInput={(e) => {
                e.target.style.height = "auto";
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
              }}
            />
            <button
              onClick={handleSubmit}
              disabled={isStreaming || !input.trim()}
              style={{
                width: 44, height: 44, borderRadius: "50%",
                background: isStreaming || !input.trim()
                  ? "rgba(99,102,241,0.2)"
                  : "linear-gradient(135deg, #6366f1, #4f46e5)",
                border: "none", color: "#fff", fontSize: 18,
                cursor: isStreaming || !input.trim() ? "not-allowed" : "pointer",
                display: "flex", alignItems: "center", justifyContent: "center",
                flexShrink: 0, transition: "background 0.2s",
              }}
            >
              ↑
            </button>
          </div>
          <div style={{ textAlign: "center", marginTop: 6, color: "#334155", fontSize: 11 }}>
            Enter to send · Shift+Enter for new line
          </div>
        </div>
      </div>

      {/* Keyframe styles */}
      <style>{`
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40% { transform: translateY(-8px); opacity: 1; }
        }
        * { box-sizing: border-box; }
        body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
      `}</style>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Avatar Canvas with error boundary
// ---------------------------------------------------------------------------
function AvatarCanvasWrapper({ lipSyncRef, isSpeaking, isThinking, onError }) {
  return (
    <AvatarErrorBoundary>
      <AvatarCanvas lipSyncRef={lipSyncRef} isSpeaking={isSpeaking} isThinking={isThinking} />
    </AvatarErrorBoundary>
  );
}

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------
export default function App() {
  const [session, setSession] = useState(null);

  if (!session) {
    return (
      <SetupScreen
        onStart={({ sessionId, cv, role, candidateName }) =>
          setSession({ sessionId, cv, role, candidateName })
        }
      />
    );
  }

  return (
    <InterviewScreen
      {...session}
      onReset={() => setSession(null)}
    />
  );
}
