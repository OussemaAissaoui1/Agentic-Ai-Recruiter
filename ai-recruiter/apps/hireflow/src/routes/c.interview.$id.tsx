import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { motion, AnimatePresence } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Mic, Video, ArrowRight, Check, Sparkles, Pause, Loader2, Square } from "lucide-react";

import {
  applications as applicationsApi,
  nlp as nlpApi,
  vision as visionApi,
  type Application,
  type Job,
} from "@/lib/api";
import {
  useApplication,
  useCreateInterview,
  useJob,
  useUpdateApplication,
} from "@/lib/queries";
import { useApp } from "@/lib/store";
import { toast } from "sonner";

import { LipSyncAudioQueue } from "@/lib/AudioLipSync";
import AvatarCanvas from "@/components/avatar/AvatarCanvas";
import { useTranscribeWS } from "@/hooks/useTranscribeWS";
import { useInterviewTurnSSE } from "@/hooks/useInterviewTurnSSE";
import { WaitingScreen } from "@/components/WaitingScreen";

export const Route = createFileRoute("/c/interview/$id")({
  head: () => ({ meta: [{ title: "AI Interview Room — HireFlow" }] }),
  component: Interview,
});

type Phase = "preflight" | "live" | "done";

type BehaviorScores = {
  engagement: number;
  confidence: number;
  composure: number;
  stress: number;
};

const DEFAULT_SCORES: BehaviorScores = {
  engagement: 60,
  confidence: 60,
  composure: 60,
  stress: 30,
};

function Interview() {
  const { id } = Route.useParams();
  const { data: application, isLoading: appLoading } = useApplication(id);
  const { data: job } = useJob(application?.job_id);

  const [phase, setPhase] = useState<Phase>("preflight");
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
  }, []);

  if (appLoading || !application) {
    return (
      <div className="fixed inset-0 z-50 grid place-items-center bg-ink text-white">
        <Loader2 className="h-6 w-6 animate-spin opacity-60" />
      </div>
    );
  }
  if (application.stage !== "approved" && application.stage !== "interviewed") {
    return <WaitingScreen application={application} />;
  }

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-ink text-white">
      <div className="absolute inset-0 bg-violet-grad opacity-20" />
      <header className="relative flex items-center justify-between border-b border-white/10 px-6 py-4">
        <div className="flex items-center gap-2">
          <div className="grid h-7 w-7 place-items-center rounded-lg bg-white/10">
            <Sparkles className="h-3.5 w-3.5" />
          </div>
          <div>
            <div className="text-xs text-white/60">AI Interview Room</div>
            <div className="text-sm font-semibold">{job?.title ?? "Interview"}</div>
          </div>
        </div>
        <Link to="/c" className="text-xs text-white/60 hover:text-white">Exit</Link>
      </header>
      <div className="relative flex-1 overflow-hidden">
        <AnimatePresence mode="wait">
          {phase === "preflight" && (
            <Preflight
              key="p"
              onReady={(s) => { streamRef.current = s; }}
              onStart={() => setPhase("live")}
            />
          )}
          {phase === "live" && (
            <Live
              key="l"
              stream={streamRef.current}
              application={application}
              job={job}
              onDone={() => setPhase("done")}
            />
          )}
          {phase === "done" && <Done key="d" applicationId={id} />}
        </AnimatePresence>
      </div>
    </div>
  );
}

// ─── Preflight ───────────────────────────────────────────────────────────────
function Preflight({
  onReady,
  onStart,
}: {
  onReady: (s: MediaStream) => void;
  onStart: () => void;
}) {
  const [mic, setMic] = useState(false);
  const [cam, setCam] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [requesting, setRequesting] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const request = useCallback(async () => {
    setRequesting(true);
    setError(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = s;
      onReady(s);
      setMic(s.getAudioTracks().length > 0);
      setCam(s.getVideoTracks().length > 0);
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        videoRef.current.play().catch(() => undefined);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRequesting(false);
    }
  }, [onReady]);

  useEffect(() => { void request(); }, [request]);

  const ready = mic && cam;

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="grid h-full place-items-center px-6">
      <div className="w-full max-w-md text-center">
        <div className="text-xs uppercase tracking-widest text-white/50">Preflight</div>
        <h1 className="font-display mt-2 text-4xl">Ready when you are.</h1>
        <p className="mt-3 text-pretty text-white/70">
          A few questions, conversational. Take your time. There's no live human watching — just the room and you.
        </p>
        <div className="mt-6 overflow-hidden rounded-2xl border border-white/10 bg-black/30">
          <video ref={videoRef} playsInline muted className="h-44 w-full bg-black/40 object-cover" />
        </div>
        <div className="mt-6 grid grid-cols-2 gap-3">
          <PreflightCheck icon={<Mic className="h-4 w-4" />} label="Microphone" ok={mic} />
          <PreflightCheck icon={<Video className="h-4 w-4" />} label="Camera" ok={cam} />
        </div>
        {error && (
          <div className="mt-4 rounded-lg border border-red-400/30 bg-red-500/10 px-3 py-2 text-xs text-red-200">
            {error}
            <button onClick={() => void request()} className="ml-2 underline hover:text-white">Retry</button>
          </div>
        )}
        <button onClick={onStart} disabled={!ready || requesting}
          className="mt-8 inline-flex items-center gap-1.5 rounded-full bg-white px-5 py-2.5 text-sm font-semibold text-foreground disabled:opacity-50">
          {requesting && <Loader2 className="h-4 w-4 animate-spin" />}
          Begin interview <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </motion.div>
  );
}

function PreflightCheck({ icon, label, ok }: { icon: React.ReactNode; label: string; ok: boolean }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/5 p-3 text-left">
      <div className="grid h-9 w-9 place-items-center rounded-lg bg-white/10">{icon}</div>
      <div>
        <div className="text-sm">{label}</div>
        <div className={`inline-flex items-center gap-1 text-xs ${ok ? "text-emerald-300" : "text-white/50"}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${ok ? "bg-emerald-300" : "bg-white/40"}`} />
          {ok ? "Ready" : "Waiting"}
        </div>
      </div>
    </div>
  );
}

// ─── Live ────────────────────────────────────────────────────────────────────
type Turn = { role: "user" | "assistant"; content: string };

/**
 * Convert the interview's alternating assistant/user history into the
 * canonical [{q, a}] transcript shape used by the recruit DB and scoring
 * agent. Pairs each `assistant` (recruiter) turn with the immediately
 * following `user` (candidate) turn. Trailing un-answered assistant turns
 * are dropped — scoring needs Q+A pairs.
 */
function historyToTranscript(
  history: Turn[],
): Array<{ q: string; a: string }> {
  const out: Array<{ q: string; a: string }> = [];
  let pending: string | null = null;
  for (const turn of history) {
    if (turn.role === "assistant") {
      pending = turn.content;
    } else if (turn.role === "user" && pending !== null) {
      out.push({ q: pending, a: turn.content });
      pending = null;
    }
  }
  return out;
}

function Live({
  stream,
  application,
  job,
  onDone,
}: {
  stream: MediaStream | null;
  application: Application | undefined;
  job: Job | undefined;
  onDone: () => void;
}) {
  const { profile } = useApp();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const visionWsRef = useRef<WebSocket | null>(null);
  const frameTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Lip-sync queue. Created once and reused across turns.
  const lipSyncRef = useRef<LipSyncAudioQueue | null>(null);
  if (!lipSyncRef.current) lipSyncRef.current = new LipSyncAudioQueue();

  const [scores, setScores] = useState<BehaviorScores>(DEFAULT_SCORES);
  const [seconds, setSeconds] = useState(0);
  const [qIdx, setQIdx] = useState(0);
  // Soft upper bound. Interview flows naturally via VAD turn-taking; this is a safety
  // ceiling so a runaway loop ends instead of going forever.
  const totalQ = 10;

  const [partial, setPartial] = useState("");        // streaming question text (before "done")
  const [question, setQuestion] = useState("");      // finalized question text
  const [questionDone, setQuestionDone] = useState(false);
  const [history, setHistory] = useState<Turn[]>([]);
  const [nlpSession, setNlpSession] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [isSpeaking, setIsSpeaking] = useState(false); // avatar TTS playing
  const [isThinking, setIsThinking] = useState(false); // LLM working before first token

  const jobRole = job?.title || "Candidate";
  const cvText = profile.cvText || "";

  // Persist the captured transcript to the recruit DB so the scoring agent
  // can score it later. Best-effort: a failure here doesn't block the
  // candidate from finishing — scoring can fall back to an inline override.
  const createInterview = useCreateInterview();
  const startedAtRef = useRef<number>(Date.now() / 1000);
  const persistGuard = useRef(false);
  const finishInterview = useCallback(
    async (finalHistory: Turn[]) => {
      if (persistGuard.current) {
        onDone();
        return;
      }
      persistGuard.current = true;
      try {
        const transcript = historyToTranscript(finalHistory);
        if (application?.id && transcript.length > 0) {
          await createInterview.mutateAsync({
            application_id: application.id,
            transcript,
            status: "completed",
            started_at: startedAtRef.current,
            ended_at: Date.now() / 1000,
          });
        }
      } catch (e) {
        // Don't block the candidate UX. The recruiter can still score via
        // an inline-override request if the persist failed.
        toast.error(
          `Couldn't save transcript: ${e instanceof Error ? e.message : String(e)}`,
        );
      } finally {
        onDone();
      }
    },
    [application?.id, createInterview, onDone],
  );

  // ── Self-view ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(() => undefined);
    }
  }, [stream]);

  // ── Timer ────────────────────────────────────────────────────────────────
  useEffect(() => {
    const t = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, []);

  // ── Vision WS + 4 fps frame upload ───────────────────────────────────────
  useEffect(() => {
    if (!stream) return;
    let cancelled = false;
    let ws: WebSocket | null = null;

    (async () => {
      try {
        const { session_id } = await visionApi.startSession({
          candidate_id: application?.candidate_email || application?.id,
        });
        if (cancelled) return;
        ws = new WebSocket(visionApi.wsUrl(session_id));
        visionWsRef.current = ws;
        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(typeof ev.data === "string" ? ev.data : "");
            if (msg && typeof msg === "object" && "scores" in msg) {
              setScores((prev) => ({ ...prev, ...(msg as { scores: Partial<BehaviorScores> }).scores }));
            } else if (msg && typeof msg === "object") {
              const merged: Partial<BehaviorScores> = {};
              for (const k of ["engagement", "confidence", "composure", "stress"] as const) {
                if (typeof (msg as Record<string, unknown>)[k] === "number") {
                  merged[k] = (msg as Record<string, number>)[k];
                }
              }
              if (Object.keys(merged).length) setScores((prev) => ({ ...prev, ...merged }));
            }
          } catch { /* ignore */ }
        };

        await new Promise<void>((resolve) => {
          if (!ws) return resolve();
          if (ws.readyState === WebSocket.OPEN) resolve();
          else ws.addEventListener("open", () => resolve(), { once: true });
        });

        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        canvas.width = 320;
        canvas.height = 240;
        frameTimerRef.current = setInterval(() => {
          if (!ws || ws.readyState !== WebSocket.OPEN) return;
          if (video.readyState < 2) return;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          canvas.toBlob(
            (blob) => {
              if (blob && ws && ws.readyState === WebSocket.OPEN) {
                blob.arrayBuffer().then((ab) => ws!.send(ab)).catch(() => undefined);
              }
            },
            "image/jpeg",
            0.6,
          );
        }, 250);
      } catch (e) {
        console.warn("[vision] disabled:", e);
      }
    })();

    return () => {
      cancelled = true;
      if (frameTimerRef.current) clearInterval(frameTimerRef.current);
      try { ws?.close(); } catch { /* noop */ }
      visionWsRef.current = null;
    };
  }, [stream, application?.candidate_email, application?.id]);

  // ── Wire LipSync activity → isSpeaking ───────────────────────────────────
  useEffect(() => {
    const q = lipSyncRef.current;
    if (!q) return;
    q.onActivityChange = (playing) => setIsSpeaking(playing);
    return () => {
      q.onActivityChange = undefined;
    };
  }, []);

  // ── NLP session bootstrap ────────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { session_id } = await nlpApi.session({ cv_text: cvText, job_role: jobRole });
        if (cancelled) return;
        setNlpSession(session_id);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        toast.error(`Couldn't start interview session: ${msg}`);
      }
    })();
    return () => { cancelled = true; };
  }, [cvText, jobRole]);

  // ── Streaming SSE turn driver ────────────────────────────────────────────
  const turn = useInterviewTurnSSE({
    lipSyncRef,
    onToken: (_inc, full) => {
      setIsThinking(false);
      setPartial(full);
    },
    onDone: (full) => {
      setQuestion(full);
      setQuestionDone(true);
      setPartial(full);
      setHistory((h) => [...h, { role: "assistant", content: full }]);
    },
    onError: (msg) => {
      setError(msg);
      setIsThinking(false);
      toast.error(`Question stream failed: ${msg}`);
    },
  });

  const askNext = useCallback(async (lastAnswer: string, currentHistory: Turn[]) => {
    if (!nlpSession) return;
    setPartial("");
    setQuestion("");
    setQuestionDone(false);
    setIsThinking(true);
    setLiveTranscript_ext("");
    await turn.run({
      sessionId: nlpSession,
      cvText,
      jobRole,
      answer: lastAnswer,
      history: currentHistory.map((t) => ["", t.content] as [string, string]),
    });
  }, [nlpSession, cvText, jobRole, turn]);

  // After session is up, kick the first question (empty answer triggers greeting).
  const greetedRef = useRef(false);
  useEffect(() => {
    if (!nlpSession || greetedRef.current) return;
    greetedRef.current = true;
    void askNext("", []);
  }, [nlpSession, askNext]);

  // ── STT WebSocket (16 kHz Int16 PCM → /ws/transcribe) ────────────────────
  // We hold a separate "external" liveTranscript so the hook & UI agree.
  const [liveTranscript, setLiveTranscript_ext] = useState("");
  const stt = useTranscribeWS({
    onPartial: (t) => setLiveTranscript_ext(t),
    onFinal: (t) => setLiveTranscript_ext(t),
    onError: (msg) => toast.error(`STT: ${msg}`),
  });

  // Auto-start mic once the avatar finishes speaking — matches legacy flow.
  useEffect(() => {
    if (!questionDone) return;
    if (isSpeaking) return;
    if (stt.isRecording) return;
    const t = setTimeout(() => { void stt.start(); }, 600);
    return () => clearTimeout(t);
  }, [questionDone, isSpeaking, stt]);

  // VAD silence → auto-send candidate's answer.
  const sendAnswer = useCallback(async (answer: string) => {
    if (!answer.trim()) return;
    stt.stop();
    setLiveTranscript_ext("");
    const newHistory = [...history, { role: "user" as const, content: answer }];
    setHistory(newHistory);
    if (qIdx + 1 >= totalQ) {
      void finishInterview(newHistory);
      return;
    }
    setQIdx((i) => i + 1);
    await askNext(answer, newHistory);
  }, [askNext, finishInterview, history, qIdx, stt, totalQ]);

  useEffect(() => {
    if (!stt.isRecording) return;
    if (stt.vadSpeaking) return;
    if (!liveTranscript.trim()) return;
    const t = setTimeout(() => { void sendAnswer(liveTranscript); }, 2000);
    return () => clearTimeout(t);
  }, [stt.isRecording, stt.vadSpeaking, liveTranscript, sendAnswer]);

  // Manual mic toggle (override).
  const toggleRecord = useCallback(() => {
    if (stt.isRecording) stt.stop();
    else void stt.start();
  }, [stt]);

  // Cleanup on unmount: close audio queue and any open sockets.
  useEffect(() => {
    return () => {
      lipSyncRef.current?.destroy();
      lipSyncRef.current = null;
      stt.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const waveBars = useMemo(() => Array.from({ length: 64 }), []);

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="grid h-full grid-rows-[auto_1fr_auto] gap-6 px-6 py-6">
      {/* Top: live indicator + timer (no question counter — flow is conversational) */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs uppercase tracking-widest text-white/50">
          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
          Live
        </div>
        <div className="flex items-center gap-3">
          <BehaviorChips scores={scores} />
          <div className="font-mono text-xs text-white/60">
            {Math.floor(seconds / 60).toString().padStart(2, "0")}:
            {(seconds % 60).toString().padStart(2, "0")}
          </div>
        </div>
      </div>

      {/* Center: avatar + question + self-view */}
      <div className="grid gap-8 lg:grid-cols-[420px_1fr]">
        <div className="overflow-hidden rounded-2xl border border-white/10 bg-black/30">
          <AvatarCanvas
            lipSyncRef={lipSyncRef}
            isSpeaking={isSpeaking}
            isThinking={isThinking}
          />
        </div>
        <div className="flex flex-col justify-center">
          <p className="font-display text-3xl leading-tight md:text-4xl">
            {partial || question || (error ? "" : " ")}
            {!questionDone && !error && (
              <span className="ml-1 inline-block h-7 w-2 animate-pulse bg-white align-middle" />
            )}
          </p>
          {error && (
            <div className="mt-3 rounded-lg border border-red-400/30 bg-red-500/10 px-3 py-2 text-xs text-red-200">
              {error}
            </div>
          )}
          {liveTranscript && (
            <div className="mt-6 rounded-xl border border-white/10 bg-white/5 p-3 text-sm text-white/80">
              <div className="text-[10px] uppercase tracking-widest text-white/40">Your answer</div>
              <div className="mt-1">{liveTranscript}</div>
              {stt.vadSpeaking && (
                <div className="mt-1 text-[10px] text-emerald-300">● speaking</div>
              )}
            </div>
          )}
          <div className="mt-6 flex items-center gap-3">
            <video ref={videoRef} playsInline muted
              className="h-20 w-32 rounded-lg border border-white/10 bg-black/40 object-cover" />
            <canvas ref={canvasRef} className="hidden" />
            <span className="text-xs text-white/40">Self-view</span>
          </div>
        </div>
      </div>

      {/* Bottom: controls */}
      <div className="flex items-center gap-4 rounded-2xl border border-white/10 bg-white/5 p-4">
        <button onClick={toggleRecord}
          className="grid h-10 w-10 place-items-center rounded-full bg-white/10"
          title={stt.isRecording ? "Stop recording" : "Start recording"}>
          {stt.isRecording ? <Square className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
        </button>
        <div className="flex flex-1 items-center gap-1">
          {waveBars.map((_, i) => (
            <motion.span key={i}
              animate={{
                height: stt.isRecording
                  ? [4, 4 + Math.abs(Math.sin(i + seconds)) * 22, 4]
                  : 4,
              }}
              transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.025 }}
              className="w-1 rounded bg-white/70"
            />
          ))}
        </div>
        <button onClick={() => stt.stop()} title="Pause"
          className="grid h-10 w-10 place-items-center rounded-full bg-white/10">
          <Pause className="h-4 w-4" />
        </button>
        <button onClick={() => { stt.stop(); void finishInterview(history); }}
          className="inline-flex items-center gap-1.5 rounded-full border border-white/15 bg-white/5 px-4 py-2 text-sm font-medium text-white/85 hover:bg-white/10"
          title="End the interview">
          End interview
        </button>
      </div>
    </motion.div>
  );
}

function BehaviorChips({ scores }: { scores: BehaviorScores }) {
  return (
    <div className="hidden gap-2 md:flex">
      {(Object.keys(scores) as Array<keyof BehaviorScores>).map((k) => (
        <div key={k}
          className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] uppercase tracking-widest text-white/70">
          {k} <span className="ml-1 font-mono text-white">{Math.round(scores[k])}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Done ────────────────────────────────────────────────────────────────────
function Done({ applicationId }: { applicationId: string }) {
  const update = useUpdateApplication();
  const nav = useNavigate();
  const updatedRef = useRef(false);

  useEffect(() => {
    if (updatedRef.current) return;
    updatedRef.current = true;
    (async () => {
      try {
        await applicationsApi.update(applicationId, { stage: "interviewed" });
      } catch (e) {
        try {
          await update.mutateAsync({ id: applicationId, body: { stage: "interviewed" } });
        } catch {
          toast.error(`Couldn't update status: ${e instanceof Error ? e.message : String(e)}`);
        }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [applicationId]);

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="grid h-full place-items-center px-6">
      <div className="max-w-md text-center">
        <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-emerald-400/20 text-emerald-300">
          <Check className="h-7 w-7" />
        </div>
        <h1 className="font-display mt-4 text-4xl">All done.</h1>
        <p className="mt-3 text-pretty text-white/70">
          Thanks for the time. The hiring team will review your responses within 48 hours. You'll get a decision and detailed feedback either way.
        </p>
        <button onClick={() => nav({ to: "/c/applications" })}
          className="mt-8 inline-flex items-center gap-1.5 rounded-full bg-white px-5 py-2.5 text-sm font-semibold text-foreground">
          Back to applications <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </motion.div>
  );
}
