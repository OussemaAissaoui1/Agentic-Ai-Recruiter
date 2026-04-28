import { useState, useEffect, useRef, useCallback } from "react";
import { useVisionPipeline } from "./vision/useVisionPipeline";
import BehavioralOverlay from "./vision/BehavioralOverlay";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const API_BASE = ""; // Empty string uses relative URLs (proxied by Vite to backend)
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
// Audio queue — plays WAV chunks sequentially without gaps
// ---------------------------------------------------------------------------
class AudioQueue {
  constructor() {
    this.ctx = null;
    this.queue = [];
    this.playing = false;
    this.onActivityChange = null; // (isPlaying: bool) => void
  }
  _getCtx() {
    if (!this.ctx || this.ctx.state === "closed") {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this.ctx.state === "suspended") {
      this.ctx.resume();
    }
    return this.ctx;
  }

  async enqueue(base64Wav) {
    const ctx = this._getCtx();
    const binary = atob(base64Wav);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    try {
      const audioBuffer = await ctx.decodeAudioData(bytes.buffer);
      this.queue.push(audioBuffer);
      if (!this.playing) this._playNext();
    } catch (e) {
      console.error("[audio] decode error", e);
    }
  }

  _playNext() {
    if (this.queue.length === 0) {
      this.playing = false;
      this.onActivityChange?.(false);
      return;
    }
    this.playing = true;
    this.onActivityChange?.(true);
    const ctx = this._getCtx();
    const buf = this.queue.shift();
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.onended = () => this._playNext();
    src.start();
  }

  clear() {
    this.queue = [];
    this.playing = false;
    this.onActivityChange?.(false);
  }
}

// ---------------------------------------------------------------------------
// Pulsing orb component
// ---------------------------------------------------------------------------
function RecruiterOrb({ isSpeaking, isThinking }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const timeRef = useRef(0);
  const amplitudeRef = useRef(0);
  const targetAmplitudeRef = useRef(0);

  useEffect(() => {
    if (isSpeaking) targetAmplitudeRef.current = 1.0;
    else if (isThinking) targetAmplitudeRef.current = 0.3;
    else targetAmplitudeRef.current = 0;
  }, [isSpeaking, isThinking]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const SIZE = canvas.width;
    const cx = SIZE / 2;
    const cy = SIZE / 2;
    const BASE_R = SIZE * 0.28;
    const NUM_BLOBS = 6;

    function draw() {
      // Smooth amplitude
      amplitudeRef.current += (targetAmplitudeRef.current - amplitudeRef.current) * 0.08;
      const amp = amplitudeRef.current;
      timeRef.current += 0.025;
      const t = timeRef.current;

      ctx.clearRect(0, 0, SIZE, SIZE);

      // Outer glow rings
      const numRings = 3;
      for (let r = numRings; r >= 1; r--) {
        const ringR = BASE_R + amp * 28 * r + Math.sin(t * 0.7 + r) * amp * 6;
        const alpha = amp * 0.08 / r;
        const grad = ctx.createRadialGradient(cx, cy, ringR * 0.6, cx, cy, ringR);
        grad.addColorStop(0, `rgba(99,102,241,${alpha})`);
        grad.addColorStop(1, `rgba(99,102,241,0)`);
        ctx.beginPath();
        ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
      }

      // Morphing blob outline
      const pts = [];
      for (let i = 0; i < NUM_BLOBS * 4; i++) {
        const angle = (i / (NUM_BLOBS * 4)) * Math.PI * 2;
        const blobIndex = i % NUM_BLOBS;
        const noise =
          Math.sin(t * 1.8 + blobIndex * 1.3) * amp * 18 +
          Math.sin(t * 2.5 + blobIndex * 0.7 + 1) * amp * 10 +
          Math.cos(t * 1.1 + blobIndex * 2.1) * amp * 7;
        const r = BASE_R + noise;
        pts.push({ x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r });
      }

      // Draw smooth blob
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 0; i < pts.length; i++) {
        const next = pts[(i + 1) % pts.length];
        const mx = (pts[i].x + next.x) / 2;
        const my = (pts[i].y + next.y) / 2;
        ctx.quadraticCurveTo(pts[i].x, pts[i].y, mx, my);
      }
      ctx.closePath();

      // Fill gradient
      const grad = ctx.createRadialGradient(cx, cy - BASE_R * 0.2, BASE_R * 0.1, cx, cy, BASE_R * 1.4);
      grad.addColorStop(0, "#818cf8");
      grad.addColorStop(0.5, "#6366f1");
      grad.addColorStop(1, "#4f46e5");
      ctx.fillStyle = grad;
      ctx.fill();

      // Inner highlight
      const hGrad = ctx.createRadialGradient(cx - BASE_R * 0.25, cy - BASE_R * 0.3, 2, cx, cy, BASE_R * 0.9);
      hGrad.addColorStop(0, "rgba(255,255,255,0.35)");
      hGrad.addColorStop(0.5, "rgba(255,255,255,0.05)");
      hGrad.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = hGrad;
      ctx.fill();

      animRef.current = requestAnimationFrame(draw);
    }

    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={260}
      height={260}
      style={{ display: "block", margin: "0 auto" }}
    />
  );
}

// ---------------------------------------------------------------------------
// Message bubble
// ---------------------------------------------------------------------------
function Message({ role, text, isStreaming, onReplay }) {
  const isRecruiter = role === "recruiter";
  return (
    <div
      style={{
        display: "flex",
        justifyContent: isRecruiter ? "flex-start" : "flex-end",
        marginBottom: 12,
      }}
    >
      <div
        style={{
          maxWidth: "72%",
          padding: "12px 16px",
          borderRadius: isRecruiter ? "4px 18px 18px 18px" : "18px 4px 18px 18px",
          background: isRecruiter
            ? "rgba(99,102,241,0.12)"
            : "rgba(255,255,255,0.07)",
          border: `1px solid ${isRecruiter ? "rgba(99,102,241,0.3)" : "rgba(255,255,255,0.1)"}`,
          color: "#e2e8f0",
          fontSize: 15,
          lineHeight: 1.6,
          letterSpacing: 0.1,
          position: "relative",
        }}
      >
        {text}
        {isStreaming && (
          <span
            style={{
              display: "inline-block",
              width: 8,
              height: 14,
              background: "#818cf8",
              marginLeft: 3,
              verticalAlign: "middle",
              borderRadius: 1,
              animation: "blink 0.8s step-end infinite",
            }}
          />
        )}
        {onReplay && (
          <button
            onClick={onReplay}
            style={{
              marginLeft: 8,
              background: "rgba(99,102,241,0.3)",
              border: "1px solid rgba(99,102,241,0.5)",
              borderRadius: 6,
              color: "#818cf8",
              fontSize: 12,
              padding: "4px 10px",
              cursor: "pointer",
              display: "inline-flex",
              alignItems: "center",
              gap: 4,
            }}
            title="Replay audio"
          >
            🔊 Play
          </button>
        )}
      </div>
    </div>
  );
} 
// ---------------------------------------------------------------------------
// Setup screen (CV + role)
// ---------------------------------------------------------------------------
function SetupScreen({ onStart }) {
  const [cv, setCv] = useState(DEFAULT_CV);
  const [role, setRole] = useState("AI Engineering Intern");
  const [candidateName, setCandidateName] = useState("Oussema Aissaoui");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/nlp/status`)
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => setStatus({ ready: false, message: "Cannot reach server" }));
  }, []);

  const handleStart = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/nlp/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cv_text: cv, job_role: role }),
      });
      const data = await res.json();
      onStart({ sessionId: data.session_id, cv, role, candidateName });
    } catch (e) {
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

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0f0f1a",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
      }}
    >
      <div style={{ width: "100%", maxWidth: 560 }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <RecruiterOrb isSpeaking={false} isThinking={false} />
          <h1 style={{ color: "#e2e8f0", fontSize: 26, margin: "16px 0 6px", fontWeight: 700 }}>
            AI Recruiter
          </h1>
          <p style={{ color: "#94a3b8", fontSize: 14, margin: 0 }}>
            Technical interview simulator
          </p>
        </div>

        {/* Status badge */}
        {status && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: 24,
              padding: "8px 14px",
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 8,
              fontSize: 13,
              color: status.ready ? "#86efac" : "#fca5a5",
            }}
          >
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: status.ready ? "#22c55e" : "#ef4444",
                flexShrink: 0,
              }}
            />
            {status.ready
              ? `Models ready · GPU ${status.gpu_free_gb}GB free`
              : status.message || "Server offline"}
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
              rows={12}
              style={inputStyle}
            />
          </div>
          <button
            onClick={handleStart}
            disabled={loading || !status?.ready}
            style={{
              padding: "13px 0",
              borderRadius: 10,
              background:
                loading || !status?.ready
                  ? "rgba(99,102,241,0.3)"
                  : "linear-gradient(135deg, #6366f1, #4f46e5)",
              border: "none",
              color: "#fff",
              fontSize: 15,
              fontWeight: 600,
              cursor: loading || !status?.ready ? "not-allowed" : "pointer",
              letterSpacing: 0.3,
              transition: "opacity 0.2s",
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
// Interview screen
// ---------------------------------------------------------------------------
function InterviewScreen({ sessionId, cv, role, candidateName, onReset }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [history, setHistory] = useState([]);

  const audioQueueRef = useRef(new AudioQueue());
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  // Vision pipeline — runs in parallel with the chat: webcam frames + 16 kHz
  // PCM stream to /ws/vision, dimension scores back. The overlay renders the
  // candidate preview + 4-D bars; markQuestion() annotates the timeline as
  // each generated question lands.
  const vision = useVisionPipeline({ enabled: true });

  // Connect vision exactly once when this screen mounts.
  useEffect(() => {
    vision.connect(sessionId);
    return () => vision.endSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Wire audio queue activity → orb state
  useEffect(() => {
    audioQueueRef.current.onActivityChange = setIsSpeaking;
  }, []);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  // Kick off with LLM-generated opening greeting on mount
  useEffect(() => {
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

      const url = `${API_BASE}/api/nlp/stream?${params.toString()}`;

      try {
        const response = await fetch(url);
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
            try { event = JSON.parse(raw); } catch { continue; }

            if (event.type === "token") {
              fullText += event.text;
              setStreamingText(fullText);
              setIsThinking(false);
            } else if (event.type === "audio") {
              audioQueueRef.current.enqueue(event.data);
            } else if (event.type === "done") {
              const finalText = fullText.trim();
              setMessages([{ role: "recruiter", text: finalText }]);
              setHistory([["hey", finalText]]);
              setStreamingText("");
              setIsStreaming(false);
              setIsThinking(false);
              // Mark this opening question on the behavioral timeline.
              try { vision.markQuestion(finalText); } catch (_e) {}
            }
          }
        }

        // Safety: if reader finished without "done" event, clean up
        if (fullText.trim()) {
          setMessages((prev) => {
            if (prev.length === 0) return [{ role: "recruiter", text: fullText.trim() }];
            return prev;
          });
          setHistory((prev) => prev.length === 0 ? [["hey", fullText.trim()]] : prev);
        }
        setStreamingText("");
        setIsStreaming(false);
        setIsThinking(false);
      } catch (e) {
        console.error("[greeting] Stream error:", e);
        setIsStreaming(false);
        setIsThinking(false);
      }
    };

    fetchGreeting();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sendMessage = useCallback(
    async (userText) => {
      if (isStreaming) return;
      if (!userText.trim()) return;

      // Add user message to chat
      setMessages((prev) => [...prev, { role: "user", text: userText }]);
      setInput("");
      setIsThinking(true);
      setIsStreaming(true);
      setStreamingText("");
      audioQueueRef.current.clear();

      // Build SSE URL
      const params = new URLSearchParams({
        session_id: sessionId,
        cv_text: cv,
        job_role: role,
        answer: userText,
        history: JSON.stringify(history),
      });

      const url = `${API_BASE}/api/nlp/stream?${params.toString()}`;

      // Abort any previous stream
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const response = await fetch(url, { signal: controller.signal });
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let fullText = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop(); // keep incomplete line

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const raw = line.slice(6).trim();
            if (!raw) continue;

            let event;
            try {
              event = JSON.parse(raw);
            } catch {
              continue;
            }

            if (event.type === "token") {
              fullText += event.text;
              setStreamingText(fullText);
              setIsThinking(false);
            } else if (event.type === "audio") {
              // Enqueue audio — plays as soon as previous chunk ends
              audioQueueRef.current.enqueue(event.data);
            } else if (event.type === "done") {
              // Finalize message
              const finalText = fullText.trim();
              setMessages((prev) => [
                ...prev,
                { role: "recruiter", text: finalText },
              ]);
              setHistory((prev) => [...prev, [userText, finalText]]);
              setStreamingText("");
              setIsStreaming(false);
              setIsThinking(false);
              inputRef.current?.focus();
              // Mark on the vision timeline so the behavioral overlay
              // shows where each question landed.
              try { vision.markQuestion(finalText); } catch (_e) {}
            } else if (event.type === "error") {
              console.error("[stream] error:", event.message);
              setMessages((prev) => [
                ...prev,
                {
                  role: "recruiter",
                  text: "⚠️ Something went wrong. Please try again.",
                },
              ]);
              setStreamingText("");
              setIsStreaming(false);
              setIsThinking(false);
            }
          }
        }
      } catch (e) {
        if (e.name !== "AbortError") {
          console.error("[stream] fetch error:", e);
          setMessages((prev) => [
            ...prev,
            { role: "recruiter", text: "⚠️ Connection error. Please try again." },
          ]);
        }
        setStreamingText("");
        setIsStreaming(false);
        setIsThinking(false);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sessionId, cv, role, history, isStreaming, messages.length]
  );

  const handleSubmit = () => {
    if (input.trim() && !isStreaming) {
      sendMessage(input.trim());
    }
  };

  const handleInputFocus = async () => {
    // Resume AudioContext on first user interaction (browser requirement)
    const ctx = audioQueueRef.current._getCtx();
    if (ctx.state === 'suspended') {
      await ctx.resume();
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
    audioQueueRef.current.clear();
    await fetch(`${API_BASE}/api/nlp/session/${sessionId}`, { method: "DELETE" });
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
    <div
      style={{
        minHeight: "100vh",
        background: "#0f0f1a",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Live vision overlay (camera preview + 4-D dimension scores) */}
      <BehavioralOverlay
        videoRef={vision.videoRef}
        canvasRef={vision.canvasRef}
        connected={vision.connected}
        isCalibrated={vision.isCalibrated}
        calibrationProgress={vision.calibrationProgress}
        dimensions={vision.dimensions}
        latencyMs={vision.latencyMs}
        error={vision.error}
      />

      {/* Top bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "14px 24px",
          borderBottom: "1px solid rgba(255,255,255,0.07)",
          background: "rgba(15,15,26,0.9)",
          backdropFilter: "blur(12px)",
          position: "sticky",
          top: 0,
          zIndex: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: "50%",
              background: "linear-gradient(135deg, #6366f1, #4f46e5)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 16,
            }}
          >
            🎤
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
            padding: "6px 14px",
            borderRadius: 7,
            background: "rgba(255,255,255,0.05)",
            border: "1px solid rgba(255,255,255,0.1)",
            color: "#94a3b8",
            fontSize: 13,
            cursor: "pointer",
          }}
        >
          New Interview
        </button>
      </div>

      {/* Orb area */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          paddingTop: 28,
          paddingBottom: 12,
          borderBottom: "1px solid rgba(255,255,255,0.05)",
        }}
      >
        <RecruiterOrb isSpeaking={isSpeaking} isThinking={isThinking} />
        <div
          style={{
            marginTop: 10,
            fontSize: 13,
            color: isThinking || isSpeaking || isStreaming ? "#818cf8" : "#475569",
            letterSpacing: 0.5,
            transition: "color 0.3s",
          }}
        >
          {orbLabel}
        </div>
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "20px 24px",
          maxWidth: 720,
          width: "100%",
          margin: "0 auto",
          boxSizing: "border-box",
        }}
      >
        {messages.map((msg, i) => (
          <Message
            key={i}
            role={msg.role}
            text={msg.text}
            isStreaming={false}
            onReplay={null}
          />
        ))}
        {streamingText && (
          <Message role="recruiter" text={streamingText} isStreaming={true} />
        )}
        {isThinking && !streamingText && (
          <div style={{ display: "flex", gap: 6, padding: "8px 0", paddingLeft: 4 }}>
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: "#6366f1",
                  animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
                }}
              />
            ))}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div
        style={{
          padding: "16px 24px",
          borderTop: "1px solid rgba(255,255,255,0.07)",
          background: "rgba(15,15,26,0.95)",
          backdropFilter: "blur(12px)",
        }}
      >
        <div
          style={{
            maxWidth: 720,
            margin: "0 auto",
            display: "flex",
            gap: 10,
            alignItems: "flex-end",
          }}
        >
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
              borderRadius: 12,
              color: "#e2e8f0",
              padding: "12px 16px",
              fontSize: 14,
              outline: "none",
              resize: "none",
              lineHeight: 1.5,
              maxHeight: 120,
              overflow: "auto",
              opacity: isStreaming ? 0.5 : 1,
              transition: "opacity 0.2s",
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
              width: 44,
              height: 44,
              borderRadius: "50%",
              background:
                isStreaming || !input.trim()
                  ? "rgba(99,102,241,0.2)"
                  : "linear-gradient(135deg, #6366f1, #4f46e5)",
              border: "none",
              color: "#fff",
              fontSize: 18,
              cursor: isStreaming || !input.trim() ? "not-allowed" : "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
              transition: "background 0.2s",
            }}
          >
            ↑
          </button>
        </div>
        <div
          style={{
            textAlign: "center",
            marginTop: 8,
            color: "#334155",
            fontSize: 12,
          }}
        >
          Enter to send · Shift+Enter for new line
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