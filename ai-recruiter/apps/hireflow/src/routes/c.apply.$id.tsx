import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Sparkles, Check, Upload, Loader2 } from "lucide-react";
import { useCreateApplication, useJob } from "@/lib/queries";
import { useApp } from "@/lib/store";
import { toast } from "sonner";
import type { Application } from "@/lib/api";

export const Route = createFileRoute("/c/apply/$id")({
  head: () => ({ meta: [{ title: "Apply — HireFlow" }] }),
  component: Apply,
});

function Apply() {
  const { id } = Route.useParams();
  const { data: job, isLoading } = useJob(id);
  const { candidateName, candidateEmail, setCandidate } = useApp();
  const create = useCreateApplication();
  const nav = useNavigate();

  const [stage, setStage] = useState<"form" | "scoring" | "result">("form");
  const [name, setName] = useState(candidateName || "");
  const [email, setEmail] = useState(candidateEmail || "");
  const [why, setWhy] = useState("I love the way your team thinks about UI craft.");
  const [file, setFile] = useState<File | null>(null);
  const [animatedFit, setAnimatedFit] = useState(0);
  const [result, setResult] = useState<Application | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (tickRef.current) clearInterval(tickRef.current);
    };
  }, []);

  // Animate fit counter once we know the result.
  useEffect(() => {
    if (stage !== "scoring") return;
    if (tickRef.current) clearInterval(tickRef.current);
    setAnimatedFit(0);
    const target = result ? Math.round(result.fit_score * 100) : 0;
    const startedAt = Date.now();
    tickRef.current = setInterval(() => {
      const t = Math.min(1, (Date.now() - startedAt) / 1500);
      const v = Math.round(t * target);
      setAnimatedFit(v);
      if (t >= 1 && result) {
        if (tickRef.current) clearInterval(tickRef.current);
        setTimeout(() => setStage("result"), 250);
      }
    }, 30);
  }, [stage, result]);

  const submit = async () => {
    if (!file) {
      toast.error("Please upload your CV.");
      return;
    }
    if (!email) {
      toast.error("Please enter your email.");
      return;
    }
    setCandidate({ name, email });
    setStage("scoring");
    try {
      const app = await create.mutateAsync({
        job_id: id,
        candidate_name: name || undefined,
        candidate_email: email || undefined,
        cv: file,
      });
      setResult(app);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
      setStage("form");
    }
  };

  if (isLoading) {
    return <div className="text-sm text-muted-foreground">Loading…</div>;
  }
  if (!job) return <div>Job not found</div>;

  return (
    <div className="mx-auto max-w-2xl space-y-8">
      <Link to="/c" className="text-sm text-muted-foreground hover:text-foreground">
        ← All roles
      </Link>
      <div>
        <div className="text-xs uppercase tracking-widest text-muted-foreground">{job.team}</div>
        <h1 className="font-display mt-1 text-3xl tracking-tight">{job.title}</h1>
        <div className="mt-1 text-sm text-muted-foreground">
          {job.location} · {job.currency || "$"}
          {job.salary_min}–{job.salary_max}k
        </div>
      </div>

      <AnimatePresence mode="wait">
        {stage === "form" && (
          <motion.div
            key="form"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-4 rounded-2xl border border-border bg-card p-6"
          >
            <Field label="Full name">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Aria Park"
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </Field>
            <Field label="Email">
              <input
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="aria@example.com"
                type="email"
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </Field>
            <Field label="Why this role">
              <textarea
                rows={4}
                value={why}
                onChange={(e) => setWhy(e.target.value)}
                className="w-full resize-none rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </Field>
            <Field label="CV">
              <label className="flex cursor-pointer items-center justify-between rounded-xl border border-dashed border-border bg-muted/30 p-4 text-sm text-muted-foreground transition hover:bg-muted/50">
                <span className="inline-flex items-center gap-2">
                  <Upload className="h-4 w-4" />
                  {file ? file.name : "Drop your CV (PDF, DOCX, TXT, image)"}
                </span>
                <input
                  type="file"
                  accept=".pdf,.docx,.txt,.png,.jpg,.jpeg"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  className="hidden"
                />
                {file && <span className="text-xs text-accent">Ready</span>}
              </label>
            </Field>
            <button
              onClick={submit}
              disabled={create.isPending}
              className="inline-flex w-full items-center justify-center gap-1.5 rounded-full bg-violet-grad py-2.5 text-sm font-semibold text-accent-foreground shadow-glow disabled:opacity-60"
            >
              {create.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
              Submit & see my fit <ArrowRight className="h-4 w-4" />
            </button>
          </motion.div>
        )}

        {stage === "scoring" && (
          <motion.div
            key="score"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="rounded-2xl border border-border bg-card p-10 text-center"
          >
            <div className="mx-auto inline-flex items-center gap-2 text-sm text-muted-foreground">
              <Sparkles className="h-4 w-4 text-accent" /> Scoring against this role…
            </div>
            <div className="mt-6 font-display text-7xl tabular-nums">{animatedFit}</div>
            <div className="mt-4 mx-auto h-1.5 w-64 overflow-hidden rounded-full bg-muted">
              <motion.div animate={{ width: `${animatedFit}%` }} className="h-full bg-violet-grad" />
            </div>
            {!result && (
              <div className="mt-4 inline-flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" /> Parsing CV and matching…
              </div>
            )}
          </motion.div>
        )}

        {stage === "result" && result && (
          <motion.div
            key="res"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4 rounded-2xl border border-border bg-card p-6"
          >
            <div className="grid h-12 w-12 place-items-center rounded-2xl bg-success/20 text-success-foreground">
              <Check className="h-6 w-6" />
            </div>
            <h3 className="font-display text-2xl">
              You're a strong match — {Math.round(result.fit_score * 100)}%
            </h3>
            {result.matched_skills?.length ? (
              <p className="text-pretty text-muted-foreground">
                You match on{" "}
                {result.matched_skills.slice(0, 3).map((s, i, arr) => (
                  <span key={s}>
                    <strong>{s}</strong>
                    {i < arr.length - 1 ? (i === arr.length - 2 ? ", and " : ", ") : ""}
                  </span>
                ))}
                .
                {result.missing_skills?.length
                  ? ` If you're invited to interview, the team may ask about ${result.missing_skills.slice(0, 2).join(" and ")}.`
                  : ""}
              </p>
            ) : (
              <p className="text-muted-foreground">Application submitted. The hiring team will review shortly.</p>
            )}
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-xl border border-border p-3">
                <div className="text-xs uppercase tracking-widest text-muted-foreground">Matched</div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {(result.matched_skills ?? []).map((s) => (
                    <span key={s} className="rounded-full bg-success/20 px-2 py-0.5 text-xs">
                      {s}
                    </span>
                  ))}
                </div>
              </div>
              <div className="rounded-xl border border-border p-3">
                <div className="text-xs uppercase tracking-widest text-muted-foreground">To strengthen</div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {(result.missing_skills ?? []).length ? (
                    (result.missing_skills ?? []).map((s) => (
                      <span key={s} className="rounded-full bg-destructive/15 px-2 py-0.5 text-xs">
                        {s}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-muted-foreground">Nothing critical</span>
                  )}
                </div>
              </div>
            </div>
            <div className="rounded-xl border border-border bg-background p-4">
              <div className="text-xs uppercase tracking-widest text-muted-foreground">What happens next</div>
              <div className="mt-1 font-medium">A recruiter will review your application</div>
              <div className="text-sm text-muted-foreground">
                You'll get a notification here as soon as they approve you for the AI interview.
              </div>
            </div>
            <button
              onClick={() => nav({ to: "/c/applications" })}
              className="inline-flex w-full items-center justify-center gap-1.5 rounded-full bg-violet-grad py-2.5 text-sm font-semibold text-accent-foreground shadow-glow"
            >
              Track my application <ArrowRight className="h-4 w-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
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
