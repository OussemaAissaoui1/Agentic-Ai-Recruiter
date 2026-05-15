import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import { MapPin, Sparkles, ArrowRight, CircleCheck } from "lucide-react";
import { useApplications, useJobs } from "@/lib/queries";
import { useApp } from "@/lib/store";
import { Skeleton } from "@/components/ui/skeleton";
import { SpotlightCard } from "@/components/motion/SpotlightCard";
import type { ApplicationStage } from "@/lib/api";

export const Route = createFileRoute("/c/")({
  head: () => ({ meta: [{ title: "Discover roles — HireFlow" }] }),
  component: Discover,
});

function Discover() {
  const [q, setQ] = useState("");
  const { data: jobs = [], isLoading } = useJobs({ status: "open" });
  const { candidateEmail } = useApp();
  const { data: myApps = [] } = useApplications(
    candidateEmail ? { candidate_email: candidateEmail } : {},
  );
  // job_id → stage of the most recent application. Used to flip the card
  // affordance on roles the candidate has already applied to.
  const appliedByJob = useMemo(() => {
    const map = new Map<string, ApplicationStage>();
    if (!candidateEmail) return map;
    const me = candidateEmail.toLowerCase();
    for (const a of myApps) {
      if (a.candidate_email?.toLowerCase() !== me) continue;
      // Keep the latest by created_at.
      const existing = map.get(a.job_id);
      if (!existing) map.set(a.job_id, a.stage);
    }
    return map;
  }, [myApps, candidateEmail]);

  const filtered = jobs.filter(
    (j) =>
      q === "" ||
      (j.title + j.team + j.location).toLowerCase().includes(q.toLowerCase()),
  );

  return (
    <div className="space-y-8">
      <div>
        <div className="text-xs uppercase tracking-widest text-muted-foreground">Find your next role</div>
        <h1 className="font-display mt-1 text-4xl tracking-tight md:text-5xl">
          Roles that <em className="italic text-accent">already know</em> they want you.
        </h1>
        <p className="mt-2 max-w-2xl text-pretty text-muted-foreground">
          Every job is matched in real time against your profile. Apply once, get an instant fit
          score, and skip generic forms.
        </p>
      </div>
      <div className="rounded-2xl border border-border bg-card p-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search roles, teams, or locations…"
          className="w-full rounded-xl bg-transparent px-3 py-3 text-sm outline-none"
        />
      </div>

      {isLoading && (
        <div className="grid gap-3 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-44 rounded-2xl" />
          ))}
        </div>
      )}

      {!isLoading && filtered.length === 0 && (
        <div className="rounded-2xl border border-dashed border-border bg-card p-12 text-center text-sm text-muted-foreground">
          No open roles right now. Check back soon.
        </div>
      )}

      <div className="grid gap-3 md:grid-cols-2">
        {filtered.map((j, i) => {
          const appliedStage = appliedByJob.get(j.id);
          return (
            <motion.div
              key={j.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
            >
              <SpotlightCard className="group lift rounded-2xl border border-border bg-card">
              <Link
                to="/c/jobs/$id"
                params={{ id: j.id }}
                aria-label={`View ${j.title} role`}
                className="block rounded-2xl p-5 outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-xs uppercase tracking-widest text-muted-foreground">{j.team}</div>
                    <div className="mt-1 font-display text-xl group-hover:text-accent">
                      {j.title}
                    </div>
                    <div className="mt-1 inline-flex items-center gap-1 text-sm text-muted-foreground">
                      <MapPin className="h-3.5 w-3.5" />
                      {j.location}
                    </div>
                  </div>
                  {appliedStage ? (
                    <span
                      className="inline-flex shrink-0 items-center gap-1 rounded-full border border-success/40 bg-success/15 px-2.5 py-1 text-[10px] font-medium uppercase tracking-widest text-success-foreground"
                      title="You've already applied to this role"
                    >
                      <CircleCheck className="h-3 w-3" />
                      {appliedStageLabel(appliedStage)}
                    </span>
                  ) : (
                    <span
                      className="inline-flex shrink-0 items-center gap-1 rounded-full border border-dashed border-accent/40 bg-accent/10 px-2.5 py-1 text-[10px] font-medium uppercase tracking-widest text-accent"
                      title="Your personal fit is computed the moment you apply"
                    >
                      <Sparkles className="h-3 w-3" />
                      See your fit
                    </span>
                  )}
                </div>
                <div className="mt-4 flex flex-wrap gap-1.5">
                  {(j.must_have ?? []).slice(0, 4).map((s) => (
                    <span key={s} className="rounded-full bg-muted px-2 py-0.5 text-[10px]">
                      {s}
                    </span>
                  ))}
                </div>
                <div className="mt-4 flex items-center justify-between">
                  <div className="text-sm text-muted-foreground">
                    {j.currency || "$"}
                    {j.salary_min}–{j.salary_max}k
                  </div>
                  <span className="inline-flex items-center gap-1 text-xs font-semibold text-accent">
                    View role <ArrowRight className="h-3.5 w-3.5 transition group-hover:translate-x-0.5" />
                  </span>
                </div>
              </Link>
              </SpotlightCard>
            </motion.div>
          );
        })}
      </div>
      <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
        <Sparkles className="h-3.5 w-3.5" /> Matches refresh as you update your profile
      </div>
    </div>
  );
}

function appliedStageLabel(s: ApplicationStage): string {
  switch (s) {
    case "applied":      return "Applied";
    case "approved":     return "Interview ready";
    case "interviewed":  return "Interview done";
    case "offer":        return "Offer";
    case "hired":        return "Hired";
    case "rejected":     return "Closed";
    default:             return "Applied";
  }
}
