/**
 * AI Recruiter Avatar Demo — Main Application
 *
 * Architecture:
 *   FastAPI ──SSE tokens──► Browser
 *   Browser feeds tokens into HeadTTS (in-browser Kokoro on WebGPU/WASM)
 *   HeadTTS emits audio + Oculus visemes (phoneme-accurate, 15 set)
 *   Three.js drives GLB morph targets viseme_* + jawOpen + body bones
 *
 * The server still streams `audio` and (legacy FFT) `visemes` SSE events,
 * but we now ignore them on the client and use HeadTTS instead.
 */

import { useState, useEffect, useRef, useCallback } from "react";
import AvatarCanvas from "./AvatarCanvas.jsx";
import { StreamingLipSync } from "./StreamingLipSync.js";

// ---------------------------------------------------------------------------
// Theme — light, professional palette
// ---------------------------------------------------------------------------
const THEME = {
  bgPage:        "linear-gradient(180deg, #f5f7fb 0%, #eaeef7 100%)",
  bgPanel:       "rgba(255,255,255,0.92)",
  bgPanelSolid:  "#ffffff",
  bgInput:       "#ffffff",
  border:        "#d9def0",
  borderSoft:    "#e6eaf2",
  text:          "#1f2547",
  textMuted:     "#5a6488",
  textFaint:     "#8b94b3",
  primary:       "#4f46e5",
  primaryDark:   "#4338ca",
  primarySoft:   "rgba(99,102,241,0.10)",
  primaryRing:   "rgba(99,102,241,0.32)",
  accent:        "#7c3aed",
  good:          "#16a34a",
  bad:           "#dc2626",
  warn:          "#d97706",
  bubbleAi:      "#eef1ff",
  bubbleAiBorder:"#dbe1f9",
  bubbleUser:    "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
  shadow:        "0 6px 24px -10px rgba(31,37,71,0.18)",
  shadowSoft:    "0 2px 12px -6px rgba(31,37,71,0.10)",
};

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
  constructor(props) { super(props); this.state = { hasError: false, error: null }; }
  static getDerivedStateFromError(error) { return { hasError: true, error }; }
  componentDidCatch(error, info) { console.error("[AvatarErrorBoundary]", error, info?.componentStack); }
  render() {
    if (this.state.hasError) {
      return <ErrorFallback error={this.state.error} resetFn={() => this.setState({ hasError: false, error: null })} />;
    }
    return this.props.children;
  }
}

function ErrorFallback({ error, resetFn }) {
  return (
    <div style={{
      padding: 24, textAlign: "center", color: THEME.bad,
      background: "rgba(220,38,38,0.06)", border: `1px solid rgba(220,38,38,0.2)`,
      borderRadius: 14, margin: 16,
    }}>
      <h3 style={{ margin: "0 0 8px" }}>3D Avatar Error</h3>
      <p style={{ fontSize: 13, color: THEME.textMuted, margin: "0 0 12px" }}>
        {error?.message || "Failed to load 3D avatar."}
      </p>
      {resetFn && (
        <button onClick={resetFn} style={{
          padding: "6px 16px", borderRadius: 8,
          background: THEME.primarySoft, border: `1px solid ${THEME.primaryRing}`,
          color: THEME.primary, cursor: "pointer", fontSize: 13, fontWeight: 600,
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
      marginBottom: 14,
    }}>
      <div style={{
        maxWidth: "78%",
        padding: "11px 16px",
        borderRadius: isRecruiter ? "4px 16px 16px 16px" : "16px 4px 16px 16px",
        background: isRecruiter ? THEME.bubbleAi : THEME.bubbleUser,
        border: isRecruiter ? `1px solid ${THEME.bubbleAiBorder}` : "none",
        color: isRecruiter ? THEME.text : "#fff",
        fontSize: 14.5,
        lineHeight: 1.55,
        letterSpacing: 0.05,
        boxShadow: isRecruiter ? "none" : THEME.shadowSoft,
      }}>
        {text}
        {isStreaming && (
          <span style={{
            display: "inline-block", width: 7, height: 13,
            background: isRecruiter ? THEME.primary : "#fff",
            marginLeft: 4, verticalAlign: "middle",
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
function SetupScreen({ onStart, ttsReady, ttsLoadPct, ttsError }) {
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
    background: THEME.bgInput,
    border: `1px solid ${THEME.border}`,
    borderRadius: 10,
    color: THEME.text,
    padding: "11px 14px",
    fontSize: 14,
    outline: "none",
    boxSizing: "border-box",
    resize: "vertical",
    transition: "border-color 0.15s, box-shadow 0.15s",
    fontFamily: "inherit",
  };

  const isReady = status?.ready;
  const avatarOk = status?.avatar_ready;
  const nlpOk = status?.nlp_ready;

  return (
    <div style={{
      minHeight: "100vh", background: THEME.bgPage, color: THEME.text,
      display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
    }}>
      <div style={{
        width: "100%", maxWidth: 580,
        background: THEME.bgPanelSolid, borderRadius: 18,
        padding: "36px 32px", boxShadow: THEME.shadow,
        border: `1px solid ${THEME.borderSoft}`,
      }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 30 }}>
          <div style={{
            width: 86, height: 86, borderRadius: 22,
            background: "linear-gradient(135deg, #6366f1, #4f46e5)",
            margin: "0 auto 14px", display: "flex",
            alignItems: "center", justifyContent: "center", fontSize: 38,
            boxShadow: "0 12px 26px -8px rgba(79,70,229,0.45)",
          }}>
            <span role="img" aria-label="recruiter">🤝</span>
          </div>
          <h1 style={{ color: THEME.text, fontSize: 26, margin: "0 0 4px", fontWeight: 700, letterSpacing: -0.3 }}>
            AI Recruiter
          </h1>
          <p style={{ color: THEME.textMuted, fontSize: 14, margin: 0 }}>
            Realtime 3D interview simulator · server-streamed TTS
          </p>
        </div>

        {/* Status badges */}
        <div style={{ display: "flex", gap: 8, marginBottom: 22, flexWrap: "wrap" }}>
          {[
            { label: "NLP", ok: nlpOk },
            { label: "Avatar", ok: avatarOk },
            { label: "GPU", ok: status?.engine_ready, extra: status?.gpu_free_gb ? `${status.gpu_free_gb}GB` : null },
            {
              label: "TTS",
              ok: status?.tts_ready,
              extra: status?.tts_ready ? "server" : null,
            },
          ].map(({ label, ok, extra }) => (
            <div key={label} style={{
              flex: 1, padding: "9px 12px",
              background: ok ? "rgba(22,163,74,0.06)" : ok === false ? "rgba(220,38,38,0.06)" : "#f1f3f9",
              border: `1px solid ${ok ? "rgba(22,163,74,0.22)" : ok === false ? "rgba(220,38,38,0.22)" : THEME.borderSoft}`,
              borderRadius: 10, fontSize: 12.5, fontWeight: 500,
              color: ok ? THEME.good : ok === false ? THEME.bad : THEME.textMuted,
              display: "flex", alignItems: "center", gap: 7,
            }}>
              <div style={{
                width: 7, height: 7, borderRadius: "50%",
                background: ok ? THEME.good : ok === false ? THEME.bad : THEME.textFaint,
                flexShrink: 0,
              }} />
              {label}{extra ? ` · ${extra}` : ""}
            </div>
          ))}
        </div>

        {statusError && (
          <div style={{
            padding: "10px 14px", marginBottom: 18, borderRadius: 10,
            background: "rgba(220,38,38,0.06)", border: `1px solid rgba(220,38,38,0.2)`,
            color: THEME.bad, fontSize: 13,
          }}>
            {statusError}
          </div>
        )}

        {/* Form */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <FormField label="Candidate name">
            <input value={candidateName} onChange={(e) => setCandidateName(e.target.value)}
                   placeholder="Enter candidate's full name" style={inputStyle} />
          </FormField>
          <FormField label="Job role">
            <input value={role} onChange={(e) => setRole(e.target.value)} style={inputStyle} />
          </FormField>
          <FormField label="Candidate CV">
            <textarea value={cv} onChange={(e) => setCv(e.target.value)} rows={9}
                      style={{ ...inputStyle, fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 12.5 }} />
          </FormField>

          <button
            onClick={handleStart}
            disabled={loading || !isReady}
            style={{
              padding: "13px 0", borderRadius: 12,
              background: loading || !isReady
                ? "#c7cae0"
                : "linear-gradient(135deg, #6366f1, #4f46e5)",
              border: "none", color: "#fff", fontSize: 15, fontWeight: 600,
              cursor: loading || !isReady ? "not-allowed" : "pointer",
              letterSpacing: 0.3, transition: "transform 0.05s, box-shadow 0.15s",
              boxShadow: loading || !isReady ? "none" : "0 8px 22px -8px rgba(79,70,229,0.55)",
            }}
            onMouseDown={(e) => { if (!loading && isReady) e.currentTarget.style.transform = "translateY(1px)"; }}
            onMouseUp={(e)   => { e.currentTarget.style.transform = "translateY(0)"; }}
          >
            {loading ? "Starting…" : "Start Interview"}
          </button>
        </div>
      </div>
    </div>
  );
}

function FormField({ label, children }) {
  return (
    <div>
      <label style={{
        color: THEME.textMuted, fontSize: 12.5, fontWeight: 600,
        display: "block", marginBottom: 6, letterSpacing: 0.2, textTransform: "uppercase",
      }}>
        {label}
      </label>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Interview Screen — 3D Avatar + Chat
// ---------------------------------------------------------------------------
function InterviewScreen({ sessionId, cv, role, candidateName, onReset, lipSyncRef, ttsReady }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [history, setHistory] = useState([]);
  const [avatarError, setAvatarError] = useState(null);

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  // Hook the global lipSync's activity → local "speaking" flag
  useEffect(() => {
    const lsq = lipSyncRef.current;
    if (!lsq) return;
    const prev = lsq.onActivityChange;
    lsq.onActivityChange = setIsSpeaking;
    return () => { lsq.onActivityChange = prev || null; };
  }, [lipSyncRef]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  // --- SSE stream handler (token-only; audio is now done in-browser) ---
  const handleSSEStream = useCallback(async (url, userText, isGreeting = false) => {
    const lipSync = lipSyncRef.current;

    try {
      const response = await fetch(url, { signal: abortRef.current?.signal });
      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullText = "";

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
          try { event = JSON.parse(raw); }
          catch { console.warn("[SSE] parse fail:", raw.slice(0, 100)); continue; }

          switch (event.type) {
            case "token":
              fullText += event.text;
              setStreamingText(fullText);
              setIsThinking(false);
              break;

            case "visemes":
              // Buffered, paired with the next `audio` chunk
              lipSync?.enqueueVisemes(event.data);
              break;

            case "audio":
              // Decode + schedule on the AudioContext timeline (chained, gap-free)
              lipSync?.enqueueAudio(event.data);
              break;

            case "done": {
              const finalText = fullText.trim();

              if (isGreeting) {
                setMessages([{ role: "recruiter", text: finalText }]);
                setHistory([["hey", finalText]]);
              } else {
                setMessages((prev) => [...prev, { role: "recruiter", text: finalText }]);
                setHistory((prev) => [...prev, [userText, finalText]]);
              }

              setStreamingText("");
              setIsStreaming(false);
              setIsThinking(false);
              inputRef.current?.focus();
              console.log(`[SSE] complete | chars=${finalText.length}`);
              return;
            }

            case "error":
              console.error("[SSE] server error:", event.message);
              setMessages((prev) => [...prev, { role: "recruiter", text: "Something went wrong. Please try again." }]);
              setStreamingText("");
              setIsStreaming(false);
              setIsThinking(false);
              return;

            default:
              // Unknown event types are simply ignored.
              break;
          }
        }
      }

      // Stream ended without `done`
      if (fullText.trim()) {
        const finalText = fullText.trim();
        setMessages((prev) => {
          if (isGreeting && prev.length === 0) return [{ role: "recruiter", text: finalText }];
          return [...prev, { role: "recruiter", text: finalText }];
        });
        if (isGreeting) {
          setHistory((prev) => prev.length === 0 ? [["hey", finalText]] : prev);
        } else {
          setHistory((prev) => [...prev, [userText, finalText]]);
        }
      }
      setStreamingText("");
      setIsStreaming(false);
      setIsThinking(false);
    } catch (e) {
      if (e.name === "AbortError") {
        console.log("[SSE] aborted");
      } else {
        console.error("[SSE] fetch error:", e);
        setMessages((prev) => [...prev, { role: "recruiter", text: "Connection error. Please try again." }]);
      }
      setStreamingText("");
      setIsStreaming(false);
      setIsThinking(false);
    }
  }, []);

  // Opening greeting — fires once on mount
  const greetingFired = useRef(false);
  useEffect(() => {
    if (greetingFired.current || !ttsReady) return;
    greetingFired.current = true;

    const controller = new AbortController();
    abortRef.current = controller;

    const fetchGreeting = async () => {
      setIsThinking(true);
      setIsStreaming(true);
      lipSyncRef.current?.beginBurst?.();
      lipSyncRef.current?.resumeAudio();
      const params = new URLSearchParams({
        session_id: sessionId, cv_text: cv, job_role: role,
        answer: "hey", history: JSON.stringify([]),
      });
      await handleSSEStream(`${API_BASE}/api/stream?${params.toString()}`, "hey", true);
    };
    fetchGreeting();

    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ttsReady]);

  // Send message
  const sendMessage = useCallback(
    async (userText) => {
      if (isStreaming || !userText.trim()) return;

      setMessages((prev) => [...prev, { role: "user", text: userText }]);
      setInput("");
      setIsThinking(true);
      setIsStreaming(true);
      setStreamingText("");
      // Begin a fresh synth burst — drops any in-flight stale audio.
      lipSyncRef.current?.beginBurst?.() ?? lipSyncRef.current?.clear();
      lipSyncRef.current?.resumeAudio();

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const params = new URLSearchParams({
        session_id: sessionId, cv_text: cv, job_role: role,
        answer: userText, history: JSON.stringify(history),
      });
      await handleSSEStream(`${API_BASE}/api/stream?${params.toString()}`, userText, false);
    },
    [sessionId, cv, role, history, isStreaming, handleSSEStream]
  );

  const handleSubmit = () => {
    if (input.trim() && !isStreaming) sendMessage(input.trim());
  };

  const handleInputFocus = () => {
    try { lipSyncRef.current?.resumeAudio(); } catch {}
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  };

  const handleReset = async () => {
    abortRef.current?.abort();
    lipSyncRef.current?.clear();
    try {
      await fetch(`${API_BASE}/api/session/${sessionId}`, { method: "DELETE" });
    } catch (e) { console.warn("[Reset] session delete failed:", e); }
    onReset();
  };

  const statusLabel = isThinking ? "Thinking…"
    : isSpeaking ? "Speaking"
    : isStreaming ? "Generating…"
    : "Listening";

  const statusColor = isSpeaking ? THEME.primary
    : isThinking ? THEME.accent
    : THEME.textMuted;

  return (
    <div style={{
      height: "100vh", background: THEME.bgPage, color: THEME.text,
      display: "flex", flexDirection: "row", overflow: "hidden",
    }}>
      {/* ═══════════════════════════════════════════════════ */}
      {/* LEFT — Avatar panel                                 */}
      {/* ═══════════════════════════════════════════════════ */}
      <div style={{
        flex: 1.4, height: "100vh", position: "relative",
        background: "linear-gradient(180deg, #f3f6fc 0%, #e6ecf6 70%, #dde4f0 100%)",
        borderRight: `1px solid ${THEME.borderSoft}`,
        minWidth: 0,
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

        {/* Status pill */}
        <div style={{
          position: "absolute", bottom: 26, left: "50%", transform: "translateX(-50%)",
          padding: "7px 16px", borderRadius: 999,
          background: "#ffffffd9", backdropFilter: "blur(8px)",
          border: `1px solid ${THEME.borderSoft}`,
          fontSize: 12.5, fontWeight: 500, letterSpacing: 0.3, pointerEvents: "none",
          color: statusColor, transition: "color 0.3s",
          boxShadow: THEME.shadowSoft,
          display: "flex", alignItems: "center", gap: 8,
        }}>
          <span style={{
            width: 7, height: 7, borderRadius: "50%", background: statusColor,
            animation: isSpeaking || isThinking ? "pulse 1.4s ease-in-out infinite" : "none",
          }} />
          {statusLabel}
        </div>

        {/* Candidate badge */}
        <div style={{
          position: "absolute", top: 18, left: 18,
          padding: "6px 14px", borderRadius: 999,
          background: "#ffffffd9", backdropFilter: "blur(8px)",
          fontSize: 12, color: THEME.textMuted, fontWeight: 500, pointerEvents: "none",
          border: `1px solid ${THEME.borderSoft}`, letterSpacing: 0.2,
          boxShadow: THEME.shadowSoft,
        }}>
          Candidate · {candidateName}
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════ */}
      {/* RIGHT — Chat panel                                  */}
      {/* ═══════════════════════════════════════════════════ */}
      <div style={{
        flex: 1, height: "100vh", display: "flex", flexDirection: "column",
        background: THEME.bgPanelSolid, minWidth: 360,
      }}>
        {/* ── Top bar ── */}
        <div style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "14px 22px",
          borderBottom: `1px solid ${THEME.borderSoft}`,
          background: THEME.bgPanel, backdropFilter: "blur(12px)",
          flexShrink: 0,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 10,
              background: "linear-gradient(135deg, #6366f1, #4f46e5)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 16, color: "#fff", fontWeight: 700,
              boxShadow: "0 6px 14px -6px rgba(79,70,229,0.5)",
            }}>
              A
            </div>
            <div>
              <div style={{ color: THEME.text, fontSize: 14, fontWeight: 600 }}>
                Alex · AI Recruiter
              </div>
              <div style={{ color: THEME.textMuted, fontSize: 12 }}>{role}</div>
            </div>
          </div>
          <button
            onClick={handleReset}
            style={{
              padding: "7px 14px", borderRadius: 9,
              background: "#fff",
              border: `1px solid ${THEME.border}`,
              color: THEME.textMuted, fontSize: 13, fontWeight: 500, cursor: "pointer",
              transition: "background 0.15s, color 0.15s",
            }}
            onMouseEnter={(e) => { e.currentTarget.style.background = THEME.primarySoft; e.currentTarget.style.color = THEME.primary; }}
            onMouseLeave={(e) => { e.currentTarget.style.background = "#fff"; e.currentTarget.style.color = THEME.textMuted; }}
          >
            New interview
          </button>
        </div>

        {/* ── Messages ── */}
        <div style={{
          flex: 1, overflowY: "auto", padding: "22px 22px",
          boxSizing: "border-box", background: THEME.bgPanelSolid,
        }}>
          {messages.length === 0 && !streamingText && !isThinking && (
            <div style={{
              padding: "36px 12px", textAlign: "center", color: THEME.textFaint, fontSize: 13,
            }}>
              Waiting for the recruiter to start…
            </div>
          )}
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
                  width: 8, height: 8, borderRadius: "50%", background: THEME.primary,
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
          borderTop: `1px solid ${THEME.borderSoft}`,
          background: THEME.bgPanel, backdropFilter: "blur(12px)",
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
                background: THEME.bgInput,
                border: `1px solid ${THEME.border}`,
                borderRadius: 12, color: THEME.text,
                padding: "12px 14px", fontSize: 14, outline: "none",
                resize: "none", lineHeight: 1.5, maxHeight: 120, overflow: "auto",
                opacity: isStreaming ? 0.55 : 1, transition: "opacity 0.2s, border-color 0.15s, box-shadow 0.15s",
                fontFamily: "inherit",
              }}
              onFocusCapture={(e) => {
                e.currentTarget.style.borderColor = THEME.primary;
                e.currentTarget.style.boxShadow = `0 0 0 3px ${THEME.primaryRing}`;
              }}
              onBlurCapture={(e) => {
                e.currentTarget.style.borderColor = THEME.border;
                e.currentTarget.style.boxShadow = "none";
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
                width: 44, height: 44, borderRadius: 12,
                background: isStreaming || !input.trim()
                  ? "#d4d8ed"
                  : "linear-gradient(135deg, #6366f1, #4f46e5)",
                border: "none", color: "#fff", fontSize: 18, fontWeight: 700,
                cursor: isStreaming || !input.trim() ? "not-allowed" : "pointer",
                display: "flex", alignItems: "center", justifyContent: "center",
                flexShrink: 0,
                boxShadow: isStreaming || !input.trim() ? "none" : "0 8px 20px -8px rgba(79,70,229,0.55)",
                transition: "background 0.2s, transform 0.05s",
              }}
              title="Send"
            >
              ↑
            </button>
          </div>
          <div style={{
            textAlign: "center", marginTop: 7, color: THEME.textFaint, fontSize: 11,
            letterSpacing: 0.2,
          }}>
            Enter to send · Shift+Enter for new line
          </div>
        </div>
      </div>

      {/* Keyframe styles */}
      <style>{`
        @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }
        @keyframes bounce {
          0%,80%,100% { transform: translateY(0); opacity:0.4; }
          40% { transform: translateY(-7px); opacity:1; }
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse {
          0%,100% { transform: scale(1); opacity:1; }
          50%     { transform: scale(1.5); opacity:0.55; }
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          color: ${THEME.text};
          background: ${THEME.bgPage};
          -webkit-font-smoothing: antialiased;
        }
        ::-webkit-scrollbar { width: 8px; height:8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(31,37,71,0.14); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(31,37,71,0.24); }
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
// Root — owns the streaming lip-sync engine. It's just an AudioContext +
// chained scheduler, so no model download / preload is required.
// ---------------------------------------------------------------------------
export default function App() {
  const [session, setSession] = useState(null);
  const lipSyncRef = useRef(null);

  if (!lipSyncRef.current) {
    // Construct lazily; AudioContext creation is cheap and starts suspended.
    lipSyncRef.current = new StreamingLipSync();
  }

  if (!session) {
    return (
      <SetupScreen
        ttsReady={true}
        ttsLoadPct={100}
        ttsError={null}
        onStart={({ sessionId, cv, role, candidateName }) => {
          // First user gesture — initialise + resume the AudioContext.
          lipSyncRef.current?.init?.().catch(() => {});
          lipSyncRef.current?.resumeAudio();
          setSession({ sessionId, cv, role, candidateName });
        }}
      />
    );
  }
  return (
    <InterviewScreen
      {...session}
      lipSyncRef={lipSyncRef}
      ttsReady={true}
      onReset={() => {
        lipSyncRef.current?.clear();
        setSession(null);
      }}
    />
  );
}
