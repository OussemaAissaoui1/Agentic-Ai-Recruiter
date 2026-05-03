/**
 * useTranscribeWS — capture mic at 16 kHz, push raw Int16 PCM frames to
 * `/ws/transcribe`, surface partial / final transcripts and server-side VAD.
 *
 * Mirrors the legacy `apps/_legacy/demo/src/app.jsx` (lines 496-629) STT
 * pipeline. The recorder uses `ScriptProcessorNode` for parity with the
 * legacy bundle (deprecated but works on Chromium and avoids the AudioWorklet
 * registration dance). Migrating to AudioWorklet is independent.
 */

import { useCallback, useEffect, useRef, useState } from "react";

interface Options {
  /** Defaults to `/ws/transcribe` (vite proxy → uvicorn). */
  wsUrl?: string;
  onPartial?: (text: string) => void;
  onFinal?: (text: string) => void;
  onError?: (msg: string) => void;
}

interface ReturnShape {
  start: () => Promise<void>;
  stop: () => void;
  isRecording: boolean;
  liveTranscript: string;
  vadSpeaking: boolean;
  /** Live AnalyserNode for visualizers (waveform / volume meter). */
  analyser: AnalyserNode | null;
  error: string | null;
}

export function useTranscribeWS(opts: Options = {}): ReturnShape {
  const url = opts.wsUrl
    ?? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/transcribe`;

  const [isRecording, setIsRecording] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState("");
  const [vadSpeaking, setVadSpeaking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const procRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  const onPartialRef = useRef(opts.onPartial);
  const onFinalRef = useRef(opts.onFinal);
  const onErrorRef = useRef(opts.onError);
  useEffect(() => { onPartialRef.current = opts.onPartial; }, [opts.onPartial]);
  useEffect(() => { onFinalRef.current = opts.onFinal; }, [opts.onFinal]);
  useEffect(() => { onErrorRef.current = opts.onError; }, [opts.onError]);

  const stop = useCallback(() => {
    if (procRef.current) {
      try { procRef.current.disconnect(); } catch { /* ignore */ }
      procRef.current = null;
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try { wsRef.current.send(JSON.stringify({ command: "stop" })); } catch { /* ignore */ }
      setTimeout(() => {
        if (wsRef.current) {
          try { wsRef.current.close(); } catch { /* ignore */ }
          wsRef.current = null;
        }
      }, 1000);
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (ctxRef.current && ctxRef.current.state !== "closed") {
      ctxRef.current.close().catch(() => {});
      ctxRef.current = null;
    }
    analyserRef.current = null;
    setIsRecording(false);
    setVadSpeaking(false);
  }, []);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const Ctor = window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new Ctor({ sampleRate: 16000 });
      ctxRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      analyserRef.current = analyser;

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "partial") {
            setLiveTranscript(data.text);
            onPartialRef.current?.(data.text);
          } else if (data.type === "final") {
            setLiveTranscript(data.text);
            onFinalRef.current?.(data.text);
          } else if (data.type === "vad") {
            setVadSpeaking(!!data.speech);
          } else if (data.type === "error") {
            const msg = String(data.message || "STT error");
            setError(msg);
            onErrorRef.current?.(msg);
          }
        } catch {
          /* ignore non-JSON */
        }
      };
      ws.onerror = () => {
        setError("websocket error");
      };

      const bufferSize = 4096;
      const proc = ctx.createScriptProcessor(bufferSize, 1, 1);
      procRef.current = proc;
      source.connect(proc);
      proc.connect(ctx.destination);

      proc.onaudioprocess = (e) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        const float32 = e.inputBuffer.getChannelData(0);
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
          const s = Math.max(-1, Math.min(1, float32[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        ws.send(int16.buffer);
      };

      setIsRecording(true);
      setLiveTranscript("");
      setError(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "mic error";
      setError(msg);
      onErrorRef.current?.(msg);
      stop();
    }
  }, [url, stop]);

  useEffect(() => () => stop(), [stop]);

  return { start, stop, isRecording, liveTranscript, vadSpeaking, analyser: analyserRef.current, error };
}
