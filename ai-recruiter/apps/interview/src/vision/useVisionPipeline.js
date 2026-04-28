// useVisionPipeline — lightweight React hook that runs the candidate's
// webcam during the interview and streams JPEG frames + 16 kHz PCM to
// /ws/vision. Returns the live 4-D dimension scores so the interview UI
// can show a small behavioral overlay alongside the chat.
//
// Sister hook to apps/behavior/src/hooks/useStressDetection.js but stripped
// down: no PDF reports here, no calibration UI bling — just stream and
// surface dimension_scores. Reports are owned by the behavior dashboard.

import { useCallback, useEffect, useRef, useState } from 'react';

const FPS = 4;                       // 4 fps is plenty for AU stats
const AUDIO_SR = 44100;
const PING_INTERVAL_MS = 10000;
const RECONNECT_DELAY_MS = 1500;

export function useVisionPipeline({ enabled = true } = {}) {
  const wsRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const audioCtxRef = useRef(null);
  const audioWorkletRef = useRef(null);
  const sendIntervalRef = useRef(null);
  const pingRef = useRef(null);
  const sessionIdRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [isCalibrated, setCalibrated] = useState(false);
  const [dimensions, setDimensions] = useState(null);
  const [latencyMs, setLatencyMs] = useState(null);
  const [error, setError] = useState(null);

  // ---------------------------------------------------------------- helpers
  const startCamera = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 }, audio: true,
    });
    videoRef.current.srcObject = stream;
    await videoRef.current.play();

    // Audio capture via Web Audio API → ScriptProcessor (works in all browsers)
    audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: AUDIO_SR,
    });
    const src = audioCtxRef.current.createMediaStreamSource(stream);
    const processor = audioCtxRef.current.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
      if (!wsRef.current || wsRef.current.readyState !== 1) return;
      const f32 = e.inputBuffer.getChannelData(0);
      // Send raw float32 PCM, base64 encoded
      const buf = new ArrayBuffer(f32.length * 4);
      new Float32Array(buf).set(f32);
      const b64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
      try {
        wsRef.current.send(JSON.stringify({
          type: 'audio_chunk', data: b64, sample_rate: AUDIO_SR,
        }));
      } catch (_e) {}
    };
    src.connect(processor);
    processor.connect(audioCtxRef.current.destination);
    audioWorkletRef.current = processor;
    return stream;
  }, []);

  const stopCamera = useCallback(() => {
    try { videoRef.current?.srcObject?.getTracks().forEach((t) => t.stop()); } catch (_e) {}
    try { audioWorkletRef.current?.disconnect(); } catch (_e) {}
    try { audioCtxRef.current?.close(); } catch (_e) {}
    audioWorkletRef.current = null;
    audioCtxRef.current = null;
  }, []);

  const sendFrame = useCallback(() => {
    const ws = wsRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!ws || ws.readyState !== 1 || !video || !canvas) return;
    if (video.readyState < 2) return;
    canvas.width = 640; canvas.height = 480;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, 640, 480);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
    try { ws.send(JSON.stringify({ type: 'video_frame', data: dataUrl })); } catch (_e) {}
  }, []);

  // ---------------------------------------------------------------- lifecycle
  const connect = useCallback(async (sessionId) => {
    if (!enabled) return null;
    setError(null);
    sessionIdRef.current = sessionId;
    try {
      // 1. Start a vision session (separate from the NLP session_id).
      const r = await fetch('/api/vision/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate_id: sessionId || 'candidate' }),
      });
      const { session_id: visionSid } = await r.json();

      // 2. Open WebSocket.
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${proto}://${window.location.host}/ws/vision?session_id=${visionSid}`);
      ws.onopen = async () => {
        setConnected(true);
        try { await startCamera(); } catch (e) { setError(`camera: ${e.message}`); }
        // Auto-start calibration as soon as camera opens.
        try { ws.send(JSON.stringify({ type: 'calibration_start' })); } catch (_e) {}
        sendIntervalRef.current = setInterval(sendFrame, 1000 / FPS);
        pingRef.current = setInterval(() => {
          try { ws.send(JSON.stringify({ type: 'ping' })); } catch (_e) {}
        }, PING_INTERVAL_MS);
      };
      ws.onmessage = (ev) => {
        let msg;
        try { msg = JSON.parse(ev.data); } catch (_e) { return; }
        if (msg.type === 'result') {
          setLatencyMs(msg.latency_ms);
          if (msg.dimension_scores) setDimensions(msg.dimension_scores);
          if (typeof msg.calibration_progress === 'number') setCalibrationProgress(msg.calibration_progress);
          if (typeof msg.is_calibrated === 'boolean') setCalibrated(msg.is_calibrated);
        } else if (msg.type === 'calibration_progress') {
          setCalibrationProgress(msg.progress);
        } else if (msg.type === 'calibration_done') {
          setCalibrated(!!msg.success);
        }
      };
      ws.onerror = () => setError('WebSocket error');
      ws.onclose = () => {
        setConnected(false);
        clearInterval(sendIntervalRef.current);
        clearInterval(pingRef.current);
      };
      wsRef.current = ws;
      return visionSid;
    } catch (e) {
      setError(`vision connect failed: ${e.message}`);
      return null;
    }
  }, [enabled, sendFrame, startCamera]);

  const markQuestion = useCallback((label) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== 1 || !label) return;
    try { ws.send(JSON.stringify({ type: 'question_marker', label })); } catch (_e) {}
  }, []);

  const endSession = useCallback(() => {
    const ws = wsRef.current;
    try { ws?.send(JSON.stringify({ type: 'session_end' })); } catch (_e) {}
    try { ws?.close(); } catch (_e) {}
    wsRef.current = null;
    setConnected(false);
    stopCamera();
  }, [stopCamera]);

  useEffect(() => () => endSession(), [endSession]);

  return {
    videoRef, canvasRef,
    connect, markQuestion, endSession,
    connected, calibrationProgress, isCalibrated,
    dimensions, latencyMs, error,
  };
}
