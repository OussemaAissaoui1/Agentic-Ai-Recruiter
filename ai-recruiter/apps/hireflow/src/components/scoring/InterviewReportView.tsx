import { motion } from "framer-motion";

import { RecommendationChip } from "./RecommendationChip";
import type { InterviewReport, TurnScore } from "@/lib/api";

function ScorePill({ value, label }: { value: number | null; label: string }) {
  const tone =
    value === null
      ? "bg-muted text-muted-foreground"
      : value >= 4
        ? "bg-success/20 text-success-foreground"
        : value >= 3
          ? "bg-accent/15 text-accent"
          : value >= 2
            ? "bg-warning/15 text-warning-foreground"
            : "bg-destructive/15 text-destructive";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium ${tone}`}
    >
      <span className="text-[10px] uppercase tracking-wide opacity-70">
        {label}
      </span>
      <span className="font-mono font-semibold">
        {value === null ? "n/a" : `${value}/5`}
      </span>
    </span>
  );
}

function TurnCard({ turn, index }: { turn: TurnScore; index: number }) {
  const failed =
    turn.technical_score === null || turn.coherence_score === null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, delay: index * 0.03 }}
      className="rounded-2xl border border-border bg-background p-5"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-xs uppercase tracking-widest text-muted-foreground">
          Turn {turn.turn_index + 1}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <ScorePill value={turn.technical_score} label="Technical" />
          <ScorePill value={turn.coherence_score} label="Coherence" />
          {failed && (
            <span className="inline-flex items-center rounded-full bg-destructive/10 px-2 py-0.5 text-[10px] italic text-destructive">
              scoring failed
            </span>
          )}
        </div>
      </div>

      <div className="mt-4 space-y-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            Question
          </div>
          <p className="mt-1 text-sm leading-relaxed">{turn.question}</p>
        </div>
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            Answer
          </div>
          <p className="mt-1 whitespace-pre-wrap text-sm leading-relaxed text-foreground/90">
            {turn.answer}
          </p>
        </div>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <div className="rounded-xl border border-border/60 bg-muted/30 p-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            Technical
          </div>
          <p className="mt-1 text-xs leading-relaxed">
            {turn.technical_rationale || "—"}
          </p>
        </div>
        <div className="rounded-xl border border-border/60 bg-muted/30 p-3">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
            Coherence
          </div>
          <p className="mt-1 text-xs leading-relaxed">
            {turn.coherence_rationale || "—"}
          </p>
        </div>
      </div>
    </motion.div>
  );
}

export function InterviewReportView({ report }: { report: InterviewReport }) {
  const generated = new Date(report.generated_at * 1000);
  const o = report.overall;

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-border bg-gradient-to-br from-background to-muted/30 p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            <div className="text-xs uppercase tracking-widest text-muted-foreground">
              Interview report
            </div>
            <h1 className="mt-1 font-display text-2xl">
              {report.candidate_name || "Unknown candidate"}
              {report.job_title && (
                <span className="text-muted-foreground"> · {report.job_title}</span>
              )}
            </h1>
            <div className="mt-1 text-xs text-muted-foreground">
              Generated {generated.toLocaleString()} · {report.model}
            </div>
          </div>
          <RecommendationChip value={o.recommendation} />
        </div>

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <div className="rounded-xl border border-border bg-background p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Technical average
            </div>
            <div className="mt-1 font-mono text-3xl font-semibold">
              {o.technical_avg.toFixed(2)}
              <span className="text-base text-muted-foreground">/5</span>
            </div>
          </div>
          <div className="rounded-xl border border-border bg-background p-4">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Coherence average
            </div>
            <div className="mt-1 font-mono text-3xl font-semibold">
              {o.coherence_avg.toFixed(2)}
              <span className="text-base text-muted-foreground">/5</span>
            </div>
          </div>
        </div>

        {o.summary && (
          <div className="mt-5">
            <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Summary
            </div>
            <p className="mt-2 text-sm leading-relaxed">{o.summary}</p>
          </div>
        )}

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Strengths
            </div>
            <ul className="mt-2 space-y-1.5">
              {(o.strengths.length ? o.strengths : ["—"]).map((s, i) => (
                <li
                  key={i}
                  className="flex items-start gap-2 text-sm leading-relaxed"
                >
                  <span className="mt-1.5 inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-success" />
                  <span>{s}</span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Concerns
            </div>
            <ul className="mt-2 space-y-1.5">
              {(o.concerns.length ? o.concerns : ["—"]).map((s, i) => (
                <li
                  key={i}
                  className="flex items-start gap-2 text-sm leading-relaxed"
                >
                  <span className="mt-1.5 inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-destructive" />
                  <span>{s}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div>
        <h2 className="font-display text-lg">Per-answer breakdown</h2>
        <p className="mt-1 text-xs text-muted-foreground">
          {report.turns.length}{" "}
          {report.turns.length === 1 ? "turn" : "turns"} scored
        </p>
        <div className="mt-4 space-y-3">
          {report.turns.map((t, i) => (
            <TurnCard key={t.turn_index} turn={t} index={i} />
          ))}
        </div>
      </div>
    </div>
  );
}
