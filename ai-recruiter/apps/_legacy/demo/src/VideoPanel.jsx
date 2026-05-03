/**
 * VideoPanel — webcam preview + behavioral analysis WebSocket.
 *
 * Lifecycle:
 *   1. POST /api/vision/session/start  → opens a vision session (returns id)
 *   2. WebSocket /ws/vision?session_id=<id>  → bidirectional channel
 *   3. <video> shows local camera (mirrored, selfie style)
 *   4. Every ~250 ms we capture a JPEG from the canvas and ship it to the WS
 *   5. The vision agent sends back per-frame dimension scores; we surface a
 *      tiny readout (calibrating / confidence / engagement) so the candidate
 *      sees the system is alive.
 *
 * Failure handling:
 *   - getUserMedia denied → small inline error, no retry loop
 *   - vision agent unavailable → keep cam preview running, hide score panel
 *   - WS close → reconnect with backoff (1.5 s → 12 s)
 */

import { useEffect, useRef, useState } from "react";

const FRAME_HZ = 4;                 // 4 frames/sec — matches /ws/vision spec
const JPEG_QUALITY = 0.6;
const MAX_RECONNECT_MS = 12_000;

const THEME = {
  bg:           "#ffffff",
  bgSoft:       "#f3f4f8",
  border:       "#e2e6ef",
  text:         "#1f2547",
  textMuted:    "#5a6488",
  accent:       "#4f46e5",
  good:         "#16a34a",
  warn:         "#d97706",
  bad:          "#dc2626",
};

export default function VideoPanel({ candidateName, sessionMark }) {
  const [camError, setCamError] = useState(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [vSessionId, setVSessionId] = useState(null);
  const [scores, setScores] = useState(null);
  const [calibrating, setCalibrating] = useState(true);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const frameTimerRef = useRef(null);
  const reconnectMsRef = useRef(1500);
  const aliveRef = useRef(true);
  const sentMarksRef = useRef(new Set());

  // ---------------------------------------------------------------------- init
  useEffect(() => {
    aliveRef.current = true;
    let mounted = true;

    (async () => {
      // 1. Webcam
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 15 } },
          audio: false, // voice demo's WS owns the mic
        });
        if (!mounted) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play().catch(() => {});
        }
      } catch (err) {
        console.error("[VideoPanel] getUserMedia failed:", err);
        setCamError(err?.message || "Camera unavailable");
        return;
      }

      // 2. Vision session
      let visionSid = null;
      try {
        const res = await fetch("/api/vision/session/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ candidate_id: candidateName || "candidate" }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        visionSid = data.session_id;
        if (mounted) setVSessionId(visionSid);
      } catch (err) {
        console.warn("[VideoPanel] vision session unavailable:", err?.message || err);
        // No vision → keep showing local cam only
        return;
      }

      // 3. WebSocket loop (with backoff reconnect)
      const connect = () => {
        if (!aliveRef.current) return;
        const url = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/vision?session_id=${encodeURIComponent(visionSid)}`;
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!aliveRef.current) { ws.close(); return; }
          setWsConnected(true);
          reconnectMsRef.current = 1500;
        };
        ws.onclose = () => {
          setWsConnected(false);
          if (!aliveRef.current) return;
          const delay = reconnectMsRef.current;
          reconnectMsRef.current = Math.min(delay * 1.6, MAX_RECONNECT_MS);
          setTimeout(connect, delay);
        };
        ws.onerror = (e) => {
          // close handler will handle reconnect
          console.warn("[VideoPanel] ws error", e?.message || "");
        };
        ws.onmessage = (ev) => {
          if (typeof ev.data !== "string") return;
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type === "dimensions" || msg.type === "scores" || msg.type === "frame_result") {
              const d = msg.data || msg.scores || msg;
              setScores({
                engagement: d.engagement,
                stress: d.stress,
                confidence: d.confidence,
                composure: d.composure ?? d.affect,
              });
              if (typeof d.calibrated !== "undefined") setCalibrating(!d.calibrated);
            } else if (msg.type === "calibration") {
              setCalibrating(!!msg.in_progress);
            }
          } catch {/* ignore non-JSON */}
        };
      };
      connect();

      // 4. Frame capture loop
      const intervalMs = Math.round(1000 / FRAME_HZ);
      frameTimerRef.current = setInterval(() => {
        const ws = wsRef.current;
        const v = videoRef.current;
        const c = canvasRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        if (!v || v.videoWidth === 0 || !c) return;
        c.width = v.videoWidth;
        c.height = v.videoHeight;
        const ctx = c.getContext("2d");
        ctx.drawImage(v, 0, 0, c.width, c.height);
        const dataUrl = c.toDataURL("image/jpeg", JPEG_QUALITY);
        try {
          ws.send(JSON.stringify({
            type: "video_frame",
            data: dataUrl,
            ts: Date.now() / 1000,
          }));
        } catch (e) { /* ignore — onclose will reconnect */ }
      }, intervalMs);
    })();

    return () => {
      mounted = false;
      aliveRef.current = false;
      if (frameTimerRef.current) clearInterval(frameTimerRef.current);
      if (wsRef.current) {
        try { wsRef.current.close(); } catch {}
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candidateName]);

  // ------------------------------------------------------- timeline question
  // Whenever the parent stamps a new question, forward it to the vision
  // session so it appears on the behavioral timeline.
  useEffect(() => {
    if (!sessionMark || !vSessionId) return;
    const key = sessionMark.key;
    if (!key || sentMarksRef.current.has(key)) return;
    sentMarksRef.current.add(key);
    fetch(`/api/vision/session/${vSessionId}/question`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: sessionMark.text || "" }),
    }).catch(() => {});
  }, [sessionMark, vSessionId]);

  // ------------------------------------------------------------------ render
  return (
    <div style={{
      flex: 1,
      display: "flex",
      flexDirection: "column",
      background: THEME.bg,
      borderBottom: `1px solid ${THEME.border}`,
      overflow: "hidden",
      position: "relative",
    }}>
      {/* Header */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "10px 16px", borderBottom: `1px solid ${THEME.border}`,
        background: THEME.bgSoft, fontSize: 12, color: THEME.textMuted,
        flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              width: 8, height: 8, borderRadius: "50%",
              background: camError ? THEME.bad : wsConnected ? THEME.good : "#cbd5e1",
              boxShadow: !camError && wsConnected ? `0 0 0 4px ${THEME.good}22` : "none",
              transition: "all .3s",
            }}
          />
          <span style={{ fontWeight: 600, color: THEME.text, letterSpacing: 0.2 }}>
            Behavioral Analysis
          </span>
        </div>
        <span style={{ fontSize: 11 }}>
          {camError ? "no camera"
            : !wsConnected ? "connecting…"
            : calibrating ? "calibrating…"
            : "live"}
        </span>
      </div>

      {/* Video / placeholder */}
      <div style={{
        flex: 1, position: "relative", background: "#000",
        display: "flex", alignItems: "center", justifyContent: "center",
        overflow: "hidden",
      }}>
        {camError ? (
          <div style={{
            color: "#fca5a5", fontSize: 13, padding: 16, textAlign: "center", lineHeight: 1.5,
          }}>
            Camera unavailable<br />
            <span style={{ color: "#fef3c7", fontSize: 11 }}>{camError}</span>
          </div>
        ) : (
          <video
            ref={videoRef}
            muted
            playsInline
            style={{
              width: "100%", height: "100%", objectFit: "cover",
              transform: "scaleX(-1)", // mirrored selfie style
              background: "#000",
            }}
          />
        )}
        <canvas ref={canvasRef} style={{ display: "none" }} />

        {/* Score chips overlay */}
        {scores && !camError && (
          <div style={{
            position: "absolute", left: 10, right: 10, bottom: 10,
            display: "flex", gap: 6, flexWrap: "wrap", justifyContent: "center",
            pointerEvents: "none",
          }}>
            <Chip label="engagement" value={scores.engagement} />
            <Chip label="confidence" value={scores.confidence} />
            <Chip label="composure"  value={scores.composure} invert />
            <Chip label="stress"     value={scores.stress} invert />
          </div>
        )}
      </div>
    </div>
  );
}

function Chip({ label, value, invert = false }) {
  if (typeof value !== "number" || isNaN(value)) return null;
  const v = Math.max(0, Math.min(1, value));
  const score = invert ? 1 - v : v;
  const hue = 120 * score; // 0=red → 120=green
  return (
    <span style={{
      fontSize: 10.5,
      letterSpacing: 0.3,
      padding: "3px 8px",
      borderRadius: 99,
      background: "rgba(0,0,0,0.55)",
      backdropFilter: "blur(8px)",
      color: `hsl(${hue}deg 75% 75%)`,
      border: `1px solid hsla(${hue}deg 60% 55% / 0.35)`,
      fontVariantNumeric: "tabular-nums",
    }}>
      {label} {(v * 100).toFixed(0)}
    </span>
  );
}
