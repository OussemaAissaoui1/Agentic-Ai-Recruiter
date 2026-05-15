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
import { Confetti } from "@/components/motion/Confetti";

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
      <div className="fixed inset-0 z-50 grid place-items-center bg-background text-foreground">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }
  if (application.stage !== "approved" && application.stage !== "interviewed") {
    return <WaitingScreen application={application} />;
  }

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-background text-foreground">
      <div aria-hidden className="absolute inset-0 -z-10 bg-aurora opacity-60" />
      <div aria-hidden className="absolute inset-0 -z-10 grid-fade opacity-40" />
      <header className="relative flex items-center justify-between border-b border-border bg-background/70 px-6 py-3 backdrop-blur-xl">
        <div className="flex items-center gap-2.5">
          <div className="relative grid h-8 w-8 place-items-center overflow-hidden rounded-lg bg-ink text-white">
            <div className="absolute inset-0 bg-violet-grad opacity-90" />
            <Sparkles className="relative h-3.5 w-3.5" />
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
              AI Interview Room
            </div>
            <div className="text-sm font-semibold">
              {job?.title ?? "Interview"}
            </div>
          </div>
        </div>
        <Link
          to="/c"
          className="press-tight rounded-full border border-border bg-card px-3 py-1.5 text-xs font-medium text-muted-foreground transition-transform hover:scale-[1.02] hover:bg-muted hover:text-foreground"
        >
          Exit
        </Link>
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
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="grid h-full place-items-center px-6"
    >
      <div className="w-full max-w-md">
        <div className="rounded-3xl border border-border bg-card/80 p-7 shadow-card-soft backdrop-blur">
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
            Preflight
          </div>
          <h1 className="font-display mt-2 text-3xl tracking-tight">
            Ready when you are.
          </h1>
          <p className="mt-2 text-pretty text-sm text-muted-foreground">
            A few questions, conversational. Take your time. There's no
            live human watching — just the room and you.
          </p>
          <div className="mt-5 overflow-hidden rounded-2xl border border-border bg-muted/40">
            <video
              ref={videoRef}
              playsInline
              muted
              className="aspect-video w-full object-cover"
            />
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2.5">
            <PreflightCheck icon={<Mic className="h-4 w-4" />} label="Microphone" ok={mic} />
            <PreflightCheck icon={<Video className="h-4 w-4" />} label="Camera" ok={cam} />
          </div>
          {error && (
            <div className="mt-4 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              {error}
              <button
                onClick={() => void request()}
                className="ml-2 underline hover:opacity-80"
              >
                Retry
              </button>
            </div>
          )}
          <button
            onClick={onStart}
            disabled={!ready || requesting}
            className="press-tight glow-pulse mt-6 inline-flex w-full items-center justify-center gap-1.5 rounded-full bg-violet-grad px-5 py-2.5 text-sm font-semibold text-accent-foreground shadow-glow transition-transform hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {requesting && <Loader2 className="h-4 w-4 animate-spin" />}
            Begin interview <ArrowRight className="h-4 w-4" />
          </button>
        </div>
        <p className="mt-4 text-center text-[10px] text-muted-foreground">
          Your camera and microphone stream stays inside this browser
          session. Nothing is recorded for replay.
        </p>
      </div>
    </motion.div>
  );
}

function PreflightCheck({
  icon,
  label,
  ok,
}: {
  icon: React.ReactNode;
  label: string;
  ok: boolean;
}) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-border bg-background p-3 text-left">
      <div
        className={`grid h-9 w-9 place-items-center rounded-lg ${
          ok ? "bg-success/15 text-success-foreground" : "bg-muted text-muted-foreground"
        }`}
      >
        {icon}
      </div>
      <div className="min-w-0">
        <div className="text-sm font-medium">{label}</div>
        <div
          className={`inline-flex items-center gap-1 text-xs ${
            ok ? "text-success-foreground" : "text-muted-foreground"
          }`}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              ok ? "bg-success animate-pulse" : "bg-muted-foreground/40"
            }`}
          />
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
  // Holds the behavioral session report sent by the vision WS in response
  // to "session_end". Captured by the WS onmessage handler below.
  const behavioralRef = useRef<Record<string, unknown> | null>(null);
  const finishInterview = useCallback(
    async (finalHistory: Turn[]) => {
      if (persistGuard.current) {
        onDone();
        return;
      }
      persistGuard.current = true;
      try {
        // Ask the vision WS to flush its aggregator and emit report_ready,
        // then wait briefly so we can attach the summary to the createInterview
        // payload. If the WS isn't open or doesn't reply in time, we just
        // persist without behavioral context — scoring still runs on the
        // transcript alone.
        const visionWs = visionWsRef.current;
        let behavioral: Record<string, unknown> | null = null;
        if (visionWs && visionWs.readyState === WebSocket.OPEN) {
          try {
            visionWs.send(JSON.stringify({ type: "session_end" }));
          } catch {
            /* ignore — fall through with behavioral=null */
          }
          const deadline = Date.now() + 3000;
          while (Date.now() < deadline && behavioralRef.current === null) {
            await new Promise((r) => setTimeout(r, 100));
          }
          behavioral = behavioralRef.current;
        }

        const transcript = historyToTranscript(finalHistory);
        if (application?.id && transcript.length > 0) {
          await createInterview.mutateAsync({
            application_id: application.id,
            transcript,
            behavioral: behavioral ?? undefined,
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
            // Capture the behavioral session summary so finishInterview can
            // attach it to the createInterview payload.
            if (
              msg && typeof msg === "object" &&
              (msg as { type?: string }).type === "report_ready"
            ) {
              const report = (msg as { report_json?: unknown }).report_json;
              behavioralRef.current =
                report && typeof report === "object"
                  ? (report as Record<string, unknown>)
                  : null;
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
  // Skipped in dev so the typed-answer panel doesn't race with VAD on ambient sound.
  useEffect(() => {
    if (import.meta.env.DEV) return;
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

  // Dev-only typed-answer panel. Removed entirely from production builds.
  const [devAnswer, setDevAnswer] = useState("");
  const submitDevAnswer = useCallback(() => {
    const text = devAnswer.trim();
    if (!text) return;
    setDevAnswer("");
    void sendAnswer(text);
  }, [devAnswer, sendAnswer]);

  const mm = Math.floor(seconds / 60).toString().padStart(2, "0");
  const ss = (seconds % 60).toString().padStart(2, "0");
  // Soft progress: how far into the planned ceiling we are. Visual only.
  const progressPct = Math.min(100, Math.round((qIdx / totalQ) * 100));

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="mx-auto grid h-full w-full max-w-6xl grid-rows-[auto_1fr_auto] gap-5 px-6 py-5"
    >
      {/* Top bar: live + timer + soft progress */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <span className="relative grid h-2 w-2 place-items-center">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success opacity-60" />
            <span className="relative h-2 w-2 rounded-full bg-success" />
          </span>
          <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
            Live
          </span>
          <span className="text-[10px] text-muted-foreground/60">·</span>
          <span className="font-mono text-xs tabular-nums text-muted-foreground">
            {mm}:{ss}
          </span>
        </div>
        <div className="flex min-w-0 flex-1 items-center justify-end gap-3">
          <div className="hidden h-1 max-w-xs flex-1 overflow-hidden rounded-full bg-muted/60 sm:block">
            <motion.div
              animate={{ width: `${progressPct}%` }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="h-full bg-violet-grad"
            />
          </div>
          <span className="hidden text-[10px] uppercase tracking-widest text-muted-foreground sm:inline">
            Question {Math.min(qIdx + 1, totalQ)} / {totalQ}
          </span>
        </div>
      </div>

      {/* Center: avatar pod + question/answer panel; behavior column on the right (desktop) */}
      <div className="grid min-h-0 gap-5 lg:grid-cols-[440px_1fr_220px]">
        {/* Avatar pod with violet halo */}
        <div className="relative">
          <div
            aria-hidden
            className="absolute -inset-2 rounded-3xl bg-violet-grad opacity-25 blur-2xl"
          />
          <div className="relative h-full overflow-hidden rounded-3xl border border-border bg-card shadow-card-soft">
            <AvatarCanvas
              lipSyncRef={lipSyncRef}
              isSpeaking={isSpeaking}
              isThinking={isThinking}
            />
            {/* Status pill overlaid bottom-left of avatar */}
            <div className="pointer-events-none absolute bottom-3 left-3">
              <span
                className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-medium uppercase tracking-widest backdrop-blur ${
                  isSpeaking
                    ? "border-accent/40 bg-accent/15 text-accent"
                    : isThinking
                      ? "border-warning/40 bg-warning/20 text-warning-foreground"
                      : "border-border/60 bg-card/70 text-muted-foreground"
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    isSpeaking
                      ? "bg-accent animate-pulse"
                      : isThinking
                        ? "bg-warning animate-pulse"
                        : "bg-muted-foreground/60"
                  }`}
                />
                {isSpeaking ? "Speaking" : isThinking ? "Thinking" : "Listening"}
              </span>
            </div>
          </div>
        </div>

        {/* Question + answer panel */}
        <div className="flex min-h-0 flex-col">
          <div className="rounded-3xl border border-border bg-card/80 p-6 shadow-card-soft backdrop-blur md:p-8">
            <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
              Interviewer
            </div>
            <p className="font-display mt-2 text-2xl leading-snug tracking-tight md:text-3xl">
              {partial || question || (error ? "" : " ")}
              {!questionDone && !error && (
                <span className="ml-1 inline-block h-6 w-1.5 animate-pulse bg-foreground align-middle" />
              )}
            </p>
            {error && (
              <div className="mt-4 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                {error}
              </div>
            )}
          </div>

          {liveTranscript && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-3 rounded-2xl border border-accent/30 bg-accent/5 p-4"
            >
              <div className="flex items-center justify-between">
                <div className="text-[10px] uppercase tracking-widest text-accent">
                  Your answer
                </div>
                {stt.vadSpeaking && (
                  <span className="inline-flex items-center gap-1 text-[10px] font-medium text-success-foreground">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-success" />
                    Speaking
                  </span>
                )}
              </div>
              <div className="mt-1.5 text-sm leading-relaxed text-foreground/90">
                {liveTranscript}
              </div>
            </motion.div>
          )}
        </div>

        {/* Behavior pod (desktop) */}
        <aside className="hidden flex-col gap-2 lg:flex">
          <div className="rounded-2xl border border-border bg-card/80 p-3 backdrop-blur">
            <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
              Live signals
            </div>
            <div className="mt-2 grid grid-cols-1 gap-2">
              {(Object.keys(scores) as Array<keyof BehaviorScores>).map((k) => (
                <BehaviorTile key={k} label={k} value={scores[k]} />
              ))}
            </div>
          </div>
          <p className="px-1 text-[10px] leading-snug text-muted-foreground">
            Behavioral signals from your camera. Used by the recruiter to
            understand your delivery — never to filter you out.
          </p>
        </aside>
      </div>

      {/* Bottom: controls + floating self-view (mobile inline) */}
      <div className="flex items-center gap-3 rounded-3xl border border-border bg-card/80 p-3 shadow-card-soft backdrop-blur">
        <button
          onClick={toggleRecord}
          className={`press-tight grid h-10 w-10 shrink-0 place-items-center rounded-full text-white shadow-glow transition-transform hover:scale-[1.04] ${
            stt.isRecording ? "bg-destructive" : "bg-violet-grad"
          }`}
          title={stt.isRecording ? "Stop recording" : "Start recording"}
        >
          {stt.isRecording ? <Square className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
        </button>
        <div className="flex min-w-0 flex-1 items-center gap-1">
          {waveBars.map((_, i) => (
            <motion.span
              key={i}
              animate={{
                height: stt.isRecording
                  ? [4, 4 + Math.abs(Math.sin(i + seconds)) * 22, 4]
                  : 4,
              }}
              transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.025 }}
              className={`w-1 rounded ${
                stt.isRecording ? "bg-accent" : "bg-muted-foreground/40"
              }`}
            />
          ))}
        </div>
        {/* Self-view inline (acts as the live mirror) */}
        <div className="hidden h-10 w-14 shrink-0 overflow-hidden rounded-lg border border-border bg-muted/40 sm:block">
          <video
            ref={videoRef}
            playsInline
            muted
            className="h-full w-full object-cover"
          />
        </div>
        <canvas ref={canvasRef} className="hidden" />
        <button
          onClick={() => stt.stop()}
          title="Pause listening"
          className="press-tight grid h-10 w-10 shrink-0 place-items-center rounded-full border border-border bg-background text-muted-foreground transition-transform hover:scale-[1.02] hover:bg-muted hover:text-foreground"
        >
          <Pause className="h-4 w-4" />
        </button>
        <button
          onClick={() => {
            stt.stop();
            void finishInterview(history);
          }}
          className="press-tight inline-flex shrink-0 items-center gap-1.5 rounded-full border border-destructive/40 bg-destructive/10 px-4 py-2 text-xs font-semibold text-destructive transition-transform hover:scale-[1.02] hover:bg-destructive/15"
          title="End the interview"
        >
          End interview
        </button>
      </div>

      {/* Mobile-only: behavior signals as a row beneath the controls */}
      <div className="-mt-2 flex flex-wrap gap-2 lg:hidden">
        {(Object.keys(scores) as Array<keyof BehaviorScores>).map((k) => (
          <BehaviorTile key={k} label={k} value={scores[k]} compact />
        ))}
      </div>

      {import.meta.env.DEV && (
        <div className="rounded-2xl border border-warning/40 bg-warning/10 p-3">
          <div className="mb-2 flex items-center gap-2 text-[10px] uppercase tracking-widest text-warning-foreground">
            <span className="rounded bg-warning/30 px-1.5 py-0.5 font-mono">DEV</span>
            Type-as-candidate (skip voice)
          </div>
          <div className="flex items-start gap-2">
            <textarea
              value={devAnswer}
              onChange={(e) => setDevAnswer(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  e.preventDefault();
                  submitDevAnswer();
                }
              }}
              placeholder="Type the candidate's answer, then press ⌘/Ctrl+Enter or click Send."
              rows={2}
              className="flex-1 resize-y rounded-lg border border-border bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:border-accent focus:outline-none"
            />
            <button
              onClick={submitDevAnswer}
              disabled={!devAnswer.trim() || !questionDone || isSpeaking}
              className="press-tight inline-flex h-9 shrink-0 items-center gap-1.5 self-end rounded-full bg-foreground px-4 text-sm font-semibold text-background disabled:opacity-50"
              title={
                !questionDone
                  ? "Wait for the question to finish"
                  : isSpeaking
                    ? "Avatar is still speaking"
                    : "Send answer"
              }
            >
              Send <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
    </motion.div>
  );
}

function BehaviorTile({
  label,
  value,
  compact = false,
}: {
  label: string;
  value: number;
  compact?: boolean;
}) {
  const tone =
    label === "stress"
      ? value < 35
        ? "text-success-foreground"
        : value < 65
          ? "text-warning-foreground"
          : "text-destructive"
      : value >= 70
        ? "text-success-foreground"
        : value >= 45
          ? "text-foreground"
          : "text-muted-foreground";
  if (compact) {
    return (
      <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-2.5 py-1 text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
        <span className={`font-mono text-foreground ${tone}`}>
          {Math.round(value)}
        </span>
      </div>
    );
  }
  return (
    <div className="rounded-xl border border-border bg-background px-3 py-2">
      <div className="text-[9px] uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <div className="mt-0.5 flex items-baseline gap-1">
        <span className={`font-mono text-base font-semibold tabular-nums ${tone}`}>
          {Math.round(value)}
        </span>
        <span className="text-[9px] text-muted-foreground">/100</span>
      </div>
      <div className="mt-1 h-1 overflow-hidden rounded-full bg-muted/60">
        <motion.div
          animate={{ width: `${Math.max(0, Math.min(100, value))}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className={`h-full ${
            label === "stress"
              ? value < 35
                ? "bg-success"
                : value < 65
                  ? "bg-warning"
                  : "bg-destructive"
              : "bg-violet-grad"
          }`}
        />
      </div>
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
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="grid h-full place-items-center px-6"
    >
      <div className="w-full max-w-md text-center">
        <div className="rounded-3xl border border-border bg-card/80 p-8 shadow-card-soft backdrop-blur">
          <div className="relative inline-block">
            <div className="grid h-14 w-14 place-items-center rounded-2xl bg-success/20 text-success-foreground">
              <motion.span
                initial={{ scale: 0, rotate: -45 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ type: "spring", stiffness: 320, damping: 14, delay: 0.1 }}
                className="inline-flex"
              >
                <Check className="h-7 w-7" />
              </motion.span>
            </div>
            <Confetti />
          </div>
          <h1 className="font-display mt-5 text-3xl tracking-tight">All done.</h1>
          <p className="mt-2 text-pretty text-sm text-muted-foreground">
            Thanks for the time. The hiring team will review your
            responses within 48 hours. You'll get a decision and detailed
            feedback either way.
          </p>
          <button
            onClick={() => nav({ to: "/c/applications" })}
            className="press-tight glow-pulse mt-6 inline-flex items-center gap-1.5 rounded-full bg-violet-grad px-5 py-2.5 text-sm font-semibold text-accent-foreground shadow-glow transition-transform hover:scale-[1.03]"
          >
            Back to applications <ArrowRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </motion.div>
  );
}
