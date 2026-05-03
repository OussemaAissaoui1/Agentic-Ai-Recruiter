import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { ArrowRight, Check } from "lucide-react";
import { useApplications, useJobs } from "@/lib/queries";
import { useApp } from "@/lib/store";
import { Skeleton } from "@/components/ui/skeleton";
import type { Application } from "@/lib/api";

export const Route = createFileRoute("/c/applications")({
  head: () => ({ meta: [{ title: "My applications — HireFlow" }] }),
  component: Applications,
});

function stageLabel(s: Application["stage"]): string {
  switch (s) {
    case "applied":
      return "Submitted";
    case "approved":
      return "Interview ready";
    case "interviewed":
      return "Interview done";
    case "offer":
      return "Offer";
    case "hired":
      return "Hired";
    case "rejected":
      return "Closed";
    default:
      return s;
  }
}

function Applications() {
  const { candidateEmail } = useApp();
  const { data: apps = [], isLoading } = useApplications(
    candidateEmail ? { candidate_email: candidateEmail } : {},
  );
  const { data: jobs = [] } = useJobs();

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-4xl tracking-tight">My applications</h1>
        {!candidateEmail && (
          <p className="mt-1 text-sm text-muted-foreground">
            Tip: complete your <Link to="/c/profile" className="underline">profile</Link> to track your applications across devices.
          </p>
        )}
      </div>

      {isLoading && (
        <div className="space-y-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-20 rounded-2xl" />
          ))}
        </div>
      )}

      {!isLoading && apps.length === 0 && (
        <div className="rounded-2xl border border-dashed border-border bg-card p-12 text-center">
          <h3 className="font-display text-xl">No applications yet.</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Browse open roles and apply — your fit score appears instantly.
          </p>
          <Link
            to="/c"
            className="mt-4 inline-flex items-center gap-1 rounded-full bg-foreground px-4 py-2 text-xs font-semibold text-background"
          >
            Discover roles <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      )}

      <div className="space-y-3">
        {apps.map((a, i) => {
          const job = jobs.find((j) => j.id === a.job_id);
          const fit = Math.round(a.fit_score * 100);
          const label = stageLabel(a.stage);
          return (
            <motion.div
              key={a.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
              className="rounded-2xl border border-border bg-card p-5"
            >
              <div className="flex items-center gap-4">
                <div className="grid h-10 w-10 place-items-center rounded-xl bg-violet-grad text-white">
                  <Check className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-semibold">{job?.title ?? "Role"}</div>
                  <div className="text-xs text-muted-foreground">
                    {job?.team ? `${job.team} · ` : ""}applied{" "}
                    {new Date(a.created_at).toLocaleDateString()}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] uppercase tracking-widest text-muted-foreground">Fit</div>
                  <div className="font-mono text-lg text-accent">{fit}</div>
                </div>
                <span className="rounded-full bg-muted px-3 py-1 text-xs">{label}</span>
                {a.stage === "approved" && (
                  <Link
                    to="/c/interview/$id"
                    params={{ id: a.id }}
                    className="inline-flex items-center gap-1 rounded-full bg-foreground px-3 py-1.5 text-xs font-semibold text-background"
                  >
                    Start <ArrowRight className="h-3.5 w-3.5" />
                  </Link>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
