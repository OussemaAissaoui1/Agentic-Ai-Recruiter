import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { ArrowRight, Sparkles, Wand2, Check, Loader2 } from "lucide-react";
import { jobs as jobsApi, type GeneratedJD } from "@/lib/api";
import { useCreateJob } from "@/lib/queries";
import { toast } from "sonner";

export const Route = createFileRoute("/app/jobs/new")({
  head: () => ({ meta: [{ title: "New job — HireFlow" }] }),
  component: NewJob,
});

const STEPS = ["Sketch", "Generate", "Calibrate", "Publish"];

function NewJob() {
  const [step, setStep] = useState(0);
  const [title, setTitle] = useState("Senior Frontend Engineer");
  const [team, setTeam] = useState("Platform");
  const [seed, setSeed] = useState(
    "Strong React/TS, design systems, ships polished UI fast. Remote EU.",
  );
  const [location, setLocation] = useState("Remote — EU");
  const [level, setLevel] = useState("Senior");
  const [generated, setGenerated] = useState<GeneratedJD | null>(null);
  const nav = useNavigate();

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <div>
        <div className="text-xs uppercase tracking-widest text-muted-foreground">Create a role</div>
        <h1 className="font-display mt-1 text-4xl tracking-tight">New job, written by AI.</h1>
      </div>

      {/* Stepper */}
      <div className="flex items-center gap-3">
        {STEPS.map((s, i) => (
          <div key={s} className="flex flex-1 items-center gap-2">
            <div
              className={`grid h-7 w-7 place-items-center rounded-full text-xs font-semibold ${
                i <= step ? "bg-violet-grad text-accent-foreground" : "bg-muted text-muted-foreground"
              }`}
            >
              {i < step ? <Check className="h-3.5 w-3.5" /> : i + 1}
            </div>
            <div className={`text-sm ${i === step ? "font-medium" : "text-muted-foreground"}`}>{s}</div>
            {i < STEPS.length - 1 && <div className={`h-px flex-1 ${i < step ? "bg-violet-grad" : "bg-border"}`} />}
          </div>
        ))}
      </div>

      {step === 0 && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4 rounded-2xl border border-border bg-card p-6"
        >
          <Field label="Role title">
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
            />
          </Field>
          <Field label="Team">
            <input
              value={team}
              onChange={(e) => setTeam(e.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
            />
          </Field>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Location">
              <input
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </Field>
            <Field label="Level">
              <input
                value={level}
                onChange={(e) => setLevel(e.target.value)}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </Field>
          </div>
          <Field label="Sketch the role (notes, vibes, must-haves)">
            <textarea
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              rows={4}
              className="w-full resize-none rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
            />
          </Field>
          <div className="flex justify-end">
            <button
              onClick={() => setStep(1)}
              className="inline-flex items-center gap-1.5 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow"
            >
              <Wand2 className="h-4 w-4" /> Generate JD
            </button>
          </div>
        </motion.div>
      )}

      {step === 1 && (
        <GenerateStep
          title={title}
          team={team}
          seed={seed}
          location={location}
          level={level}
          onDone={(jd) => {
            setGenerated(jd);
          }}
          onNext={() => setStep(2)}
        />
      )}

      {step === 2 && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4 rounded-2xl border border-border bg-card p-6"
        >
          <h3 className="font-display text-xl">Calibrate scoring</h3>
          {[
            { k: "Skills weight", v: 35 },
            { k: "Experience weight", v: 25 },
            { k: "Culture fit weight", v: 20 },
            { k: "Trajectory weight", v: 20 },
          ].map((s) => (
            <div key={s.k} className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>{s.k}</span>
                <span className="font-mono text-muted-foreground">{s.v}%</span>
              </div>
              <input type="range" defaultValue={s.v} className="w-full accent-[oklch(0.7_0.2_295)]" />
            </div>
          ))}
          <div className="flex justify-between pt-3">
            <button
              onClick={() => setStep(1)}
              className="text-sm text-muted-foreground hover:text-foreground"
            >
              ← Back
            </button>
            <button
              onClick={() => setStep(3)}
              className="inline-flex items-center gap-1.5 rounded-full bg-foreground px-4 py-2 text-sm font-semibold text-background"
            >
              Continue <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </motion.div>
      )}

      {step === 3 && (
        <PublishStep
          title={title}
          team={team}
          location={location}
          level={level}
          jd={generated}
          onPublished={() => nav({ to: "/app/jobs" })}
        />
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs uppercase tracking-widest text-muted-foreground">{label}</span>
      <div className="mt-1.5">{children}</div>
    </label>
  );
}

function GenerateStep({
  title,
  team,
  seed,
  location,
  level,
  onDone,
  onNext,
}: {
  title: string;
  team: string;
  seed: string;
  location: string;
  level: string;
  onDone: (jd: GeneratedJD) => void;
  onNext: () => void;
}) {
  const [text, setText] = useState("");
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  useEffect(() => {
    const ac = new AbortController();
    abortRef.current = ac;
    setText("");
    setDone(false);
    setError(null);
    let buf = "";
    let finalJd: GeneratedJD | null = null;

    (async () => {
      try {
        for await (const ev of jobsApi.generateStream(
          { title, team, seed, location, level },
          { signal: ac.signal },
        )) {
          if (typeof ev.token === "string") {
            buf += ev.token;
            setText(buf);
          }
          if (ev.jd && typeof ev.jd === "object") {
            finalJd = ev.jd as GeneratedJD;
          }
        }
        if (finalJd) {
          setText(finalJd.description || buf);
          onDoneRef.current(finalJd);
        } else if (buf) {
          // No structured payload; synthesize from streamed text.
          onDoneRef.current({ description: buf, must_have: [], nice_to_have: [] });
        }
        setDone(true);
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        toast.error(`JD generation failed: ${msg}`);
      }
    })();

    return () => ac.abort();
  }, [title, team, seed, location, level]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4 rounded-2xl border border-border bg-card p-6"
    >
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Sparkles className="h-4 w-4 text-accent" />
        <span>{done ? "Generated." : "Generating with HireFlow AI…"}</span>
      </div>
      <pre className="max-h-[420px] overflow-y-auto whitespace-pre-wrap rounded-xl bg-muted/40 p-5 text-sm leading-relaxed">
        {text || (error ? "" : " ")}
        {!done && !error && <span className="ml-0.5 inline-block h-3 w-1.5 animate-pulse bg-accent align-middle" />}
      </pre>
      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-xs text-destructive">
          {error}
        </div>
      )}
      <div className="flex justify-end">
        <button
          disabled={!done}
          onClick={onNext}
          className="inline-flex items-center gap-1.5 rounded-full bg-foreground px-4 py-2 text-sm font-semibold text-background disabled:opacity-50"
        >
          Looks good <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </motion.div>
  );
}

function PublishStep({
  title,
  team,
  location,
  level,
  jd,
  onPublished,
}: {
  title: string;
  team: string;
  location: string;
  level: string;
  jd: GeneratedJD | null;
  onPublished: () => void;
}) {
  const create = useCreateJob();

  const publish = async () => {
    try {
      await create.mutateAsync({
        title,
        team,
        location,
        work_mode: "remote",
        level,
        employment_type: "Full-time",
        salary_min: 90,
        salary_max: 160,
        currency: "$",
        description: jd?.description ?? "",
        must_have: jd?.must_have ?? [],
        nice_to_have: jd?.nice_to_have ?? [],
        status: "open",
      });
      toast.success("Job published.");
      onPublished();
    } catch (e) {
      toast.error(`Publish failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4 rounded-2xl border border-border bg-card p-6"
    >
      <div className="grid h-12 w-12 place-items-center rounded-2xl bg-success/20 text-success-foreground">
        <Check className="h-6 w-6" />
      </div>
      <h3 className="font-display text-2xl">Ready to publish.</h3>
      <p className="text-muted-foreground">
        Your job is calibrated and the AI sourcing agent is queued. Candidates will start arriving
        within minutes.
      </p>
      <div className="flex gap-2">
        <button
          disabled={create.isPending}
          onClick={publish}
          className="inline-flex items-center gap-2 rounded-full bg-violet-grad px-5 py-2 text-sm font-semibold text-accent-foreground shadow-glow disabled:opacity-60"
        >
          {create.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
          Publish & view jobs
        </button>
        <button className="rounded-full border border-border px-5 py-2 text-sm font-semibold">
          Save draft
        </button>
      </div>
    </motion.div>
  );
}
