/**
 * useInterviewTurnSSE — drive an interview turn via Server-Sent Events.
 *
 * Connects to /api/nlp/stream and dispatches:
 *   - token   → onToken(text)               (incremental question text)
 *   - audio   → lipSync.enqueue(b64, vis?)  (TTS chunk → speakers + analyser)
 *   - visemes → cached for next audio chunk (server-side viseme schedule)
 *   - done    → onDone(fullText)
 *   - error   → onError(msg)
 *
 * The HireFlow NLP backend currently only emits {token,audio,done,error};
 * server-side visemes are silently ignored if absent — the AudioLipSync
 * client-side FFT path covers that case.
 */

import { useCallback, useEffect, useRef } from "react";
import type { LipSyncAudioQueue, VisemeFrame } from "@/lib/AudioLipSync";

interface Options {
  lipSyncRef: React.RefObject<LipSyncAudioQueue | null>;
  onToken?: (incrementalText: string, fullText: string) => void;
  onDone?: (fullText: string) => void;
  onError?: (message: string) => void;
}

interface TurnParams {
  sessionId: string;
  cvText: string;
  jobRole: string;
  answer: string;
  history: Array<[string, string]>;
}

export function useInterviewTurnSSE({ lipSyncRef, onToken, onDone, onError }: Options) {
  const abortRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    if (abortRef.current) {
      try { abortRef.current.abort(); } catch { /* ignore */ }
      abortRef.current = null;
    }
  }, []);

  const run = useCallback(async (params: TurnParams) => {
    cancel();
    const ac = new AbortController();
    abortRef.current = ac;

    const qs = new URLSearchParams({
      session_id: params.sessionId,
      cv_text: params.cvText,
      job_role: params.jobRole,
      answer: params.answer,
      history: JSON.stringify(params.history),
    });
    const url = `/api/nlp/stream?${qs.toString()}`;

    let fullText = "";
    let pendingVisemes: VisemeFrame[] | null = null;

    try {
      const res = await fetch(url, { signal: ac.signal });
      if (!res.ok || !res.body) {
        throw new Error(`server returned ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          let event: { type: string; [k: string]: unknown };
          try { event = JSON.parse(raw); } catch { continue; }

          switch (event.type) {
            case "token": {
              const t = String(event.text ?? "");
              fullText += t;
              onToken?.(t, fullText);
              break;
            }
            case "audio": {
              const lipSync = lipSyncRef.current;
              if (lipSync && typeof event.data === "string") {
                void lipSync.enqueue(event.data, pendingVisemes);
              }
              pendingVisemes = null;
              break;
            }
            case "visemes": {
              if (Array.isArray(event.data)) {
                pendingVisemes = event.data as VisemeFrame[];
              }
              break;
            }
            case "done":
              onDone?.(fullText.trim());
              return;
            case "error":
              onError?.(String(event.message || "stream error"));
              return;
          }
        }
      }
      // stream ended without explicit done
      if (fullText.trim()) onDone?.(fullText.trim());
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      onError?.((e as Error).message);
    } finally {
      if (abortRef.current === ac) abortRef.current = null;
    }
  }, [cancel, lipSyncRef, onToken, onDone, onError]);

  useEffect(() => () => cancel(), [cancel]);

  return { run, cancel };
}
