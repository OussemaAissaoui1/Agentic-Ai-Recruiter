import { useRef, useCallback, useEffect, useState } from 'react';

const WS_RECONNECT_DELAY = 1500;
const MAX_RECONNECT_ATTEMPTS = 20;
const WS_PING_INTERVAL = 10000; // 10s keepalive ping

// Behavior dashboard talks to /api/vision/* under the unified app.
const API_BASE = '/api/vision';

export function useStressDetection(wsUrl) {
  const wsRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const sendIntervalRef = useRef(null);
  const audioContextRef = useRef(null);
  const audioProcessorRef = useRef(null);
  const streamRef = useRef(null);
  const streamingRef = useRef(false);
  const reconnectAttempts = useRef(0);
  const reconnectTimerRef = useRef(null);
  const pingIntervalRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [latestResult, setLatestResult] = useState(null);
  const [fps, setFps] = useState(8);
  const [stats, setStats] = useState({
    framesSent: 0,
    framesAnalyzed: 0,
    stressedCount: 0,
    latencyMs: 0,
    avgStressProb: 0,
    history: [],
  });

  // ── New state for multi-dimension system ──
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [dimensionScores, setDimensionScores] = useState(null);
  const [notableMoments, setNotableMoments] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [reportData, setReportData] = useState(null);
  const [sessionStats, setSessionStats] = useState(null);

  const connectWs = useCallback((wsUrlWithSession) => {
    const targetUrl = wsUrlWithSession || wsUrl;
    // Don't connect if already open or mid-connect
    if (
      wsRef.current?.readyState === WebSocket.OPEN ||
      wsRef.current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    // Clean up any previous socket
    if (wsRef.current) {
      try { wsRef.current.close(); } catch (_) {}
      wsRef.current = null;
    }

    // Clear any pending reconnect timer
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    const ws = new WebSocket(targetUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      reconnectAttempts.current = 0;
      console.log('[WS] Connected');

      // Start keepalive pings
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, WS_PING_INTERVAL);
    };
    ws.onclose = () => {
      setConnected(false);
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }
      console.log('[WS] Closed. streaming=', streamingRef.current);
      if (streamingRef.current && reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts.current += 1;
        const delay = Math.min(WS_RECONNECT_DELAY * reconnectAttempts.current, 8000);
        console.log(`[WS] Reconnecting attempt ${reconnectAttempts.current} in ${delay}ms...`);
        reconnectTimerRef.current = setTimeout(() => connectWs(targetUrl), delay);
      }
    };
    ws.onerror = (e) => {
      console.warn('[WS] Error:', e);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // ── Handle new message types ──
      if (data.type === 'calibration_progress') {
        setCalibrationProgress(data.progress ?? 0);
        return;
      }
      if (data.type === 'calibration_done') {
        setIsCalibrated(data.success ?? false);
        setCalibrationProgress(1.0);
        return;
      }
      if (data.type === 'notable_moment') {
        setNotableMoments((prev) => {
          const next = [...prev, data].slice(-50);
          return next;
        });
        return;
      }
      if (data.type === 'report_ready') {
        setReportData(data.report_json ?? null);
        return;
      }
      if (data.type === 'question_marked' || data.type === 'pong') {
        return;
      }

      if (data.type !== 'result') return;

      const latency = data.latency_ms ?? 0;

      // Update dimension scores
      if (data.dimension_scores) {
        setDimensionScores(data.dimension_scores);
      }
      if (data.calibration_progress != null) {
        setCalibrationProgress(data.calibration_progress);
      }
      if (data.is_calibrated != null) {
        setIsCalibrated(data.is_calibrated);
      }
      if (data.session_stats) {
        setSessionStats(data.session_stats);
      }

      setLatestResult(data);
      setStats((prev) => {
        const stressProb = data.fused?.probabilities?.stressed ?? data.visual?.probabilities?.stressed ?? 0;
        const isStressed = (data.fused?.label ?? data.visual?.label) === 'stressed';
        const newHistory = [...prev.history.slice(-59), { t: Date.now(), stress: stressProb }];
        return {
          framesSent: prev.framesSent,
          framesAnalyzed: prev.framesAnalyzed + 1,
          stressedCount: prev.stressedCount + (isStressed ? 1 : 0),
          latencyMs: latency,
          avgStressProb:
            (prev.avgStressProb * prev.framesAnalyzed + stressProb) /
            (prev.framesAnalyzed + 1),
          history: newHistory,
        };
      });
    };
  }, [wsUrl]);

  const sendFrame = useCallback(() => {
    const ws = wsRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (!video || video.readyState < 2) return;

    const ctx = canvas.getContext('2d');
    canvas.width = 640;
    canvas.height = 480;
    ctx.drawImage(video, 0, 0, 640, 480);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.7);

    ws.send(JSON.stringify({ type: 'video_frame', data: dataUrl, _sendTime: Date.now() }));
    setStats((prev) => ({ ...prev, framesSent: prev.framesSent + 1 }));
  }, []);

  const startAudio = useCallback(async (stream) => {
    try {
      const ac = new AudioContext({ sampleRate: 16000 });
      await ac.resume();

      // Load AudioWorklet processor (replaces deprecated ScriptProcessorNode)
      await ac.audioWorklet.addModule('/audio-processor.js');

      const source = ac.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(ac, 'audio-capture-processor');

      workletNode.port.onmessage = (e) => {
        if (e.data.type !== 'audio') return;
        if (wsRef.current?.readyState !== WebSocket.OPEN) return;

        const merged = e.data.samples; // Float32Array
        const bytes = new Uint8Array(merged.buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);

        wsRef.current.send(
          JSON.stringify({ type: 'audio_chunk', data: btoa(binary), sample_rate: ac.sampleRate })
        );
      };

      source.connect(workletNode);
      workletNode.connect(ac.destination);
      audioContextRef.current = ac;
      audioProcessorRef.current = workletNode;
    } catch (err) {
      console.warn('Audio capture failed:', err);
    }
  }, []);

  // ── New outbound message senders ──
  const startCalibration = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'calibration_start' }));
      setCalibrationProgress(0);
      setIsCalibrated(false);
    }
  }, []);

  const completeCalibration = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'calibration_complete' }));
    }
  }, []);

  const markQuestion = useCallback((label) => {
    if (wsRef.current?.readyState === WebSocket.OPEN && label) {
      wsRef.current.send(JSON.stringify({ type: 'question_marker', label }));
    }
  }, []);

  const endSession = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'session_end' }));
    }
  }, []);

  const start = useCallback(async (enableAudio = true) => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: enableAudio,
    });
    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }

    streamingRef.current = true;
    setStreaming(true);
    reconnectAttempts.current = 0;

    // Create session first
    let sid = null;
    try {
      const res = await fetch(`${API_BASE}/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      sid = data.session_id;
      setSessionId(sid);
    } catch (err) {
      console.warn('Failed to create session:', err);
    }

    // Connect WebSocket with session_id
    const wsUrlWithSession = sid ? `${wsUrl}?session_id=${sid}` : wsUrl;
    connectWs(wsUrlWithSession);

    // Wait for WS (with timeout)
    await new Promise((resolve) => {
      let elapsed = 0;
      const check = setInterval(() => {
        elapsed += 100;
        if (wsRef.current?.readyState === WebSocket.OPEN || elapsed > 5000) {
          clearInterval(check);
          resolve();
        }
      }, 100);
    });

    sendIntervalRef.current = setInterval(sendFrame, 1000 / fps);

    if (enableAudio && stream.getAudioTracks().length > 0) {
      await startAudio(stream);
    }

    // Auto-start calibration
    startCalibration();
  }, [connectWs, sendFrame, startAudio, fps, wsUrl, startCalibration]);

  const stop = useCallback(() => {
    endSession();
    streamingRef.current = false;
    setStreaming(false);
    if (sendIntervalRef.current) clearInterval(sendIntervalRef.current);
    if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
    if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    if (audioProcessorRef.current) audioProcessorRef.current.disconnect();
    if (audioContextRef.current) audioContextRef.current.close();
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    if (wsRef.current) wsRef.current.close();
    audioContextRef.current = null;
    audioProcessorRef.current = null;
    streamRef.current = null;
    pingIntervalRef.current = null;
    reconnectTimerRef.current = null;
  }, [endSession]);

  // Update send interval when fps changes
  useEffect(() => {
    if (streaming && sendIntervalRef.current) {
      clearInterval(sendIntervalRef.current);
      sendIntervalRef.current = setInterval(sendFrame, 1000 / fps);
    }
  }, [fps, streaming, sendFrame]);

  // Cleanup on unmount
  useEffect(() => () => stop(), [stop]);

  return {
    videoRef,
    canvasRef,
    connected,
    streaming,
    latestResult,
    stats,
    fps,
    setFps,
    start,
    stop,
    // New exports
    calibrationProgress,
    isCalibrated,
    dimensionScores,
    notableMoments,
    sessionId,
    reportData,
    sessionStats,
    startCalibration,
    completeCalibration,
    markQuestion,
    endSession,
  };
}
