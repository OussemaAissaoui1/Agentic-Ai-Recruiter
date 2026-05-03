import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";
import { useJobs } from "@/lib/queries";
import { Skeleton } from "@/components/ui/skeleton";
import type { Job } from "@/lib/api";

export const Route = createFileRoute("/app/jobs/")({
  head: () => ({ meta: [{ title: "Jobs — HireFlow" }] }),
  component: JobsList,
});

function JobsList() {
  const { data: jobs = [], isLoading, isError } = useJobs();
  const open = jobs.filter((j) => j.status === "open").length;

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="font-display text-4xl tracking-tight">Jobs</h1>
          <p className="mt-1 text-muted-foreground">
            {isLoading ? "Loading…" : `${jobs.length} roles, ${open} open`}
          </p>
        </div>
        <Link
          to="/app/jobs/new"
          className="inline-flex items-center gap-1.5 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow"
        >
          <Sparkles className="h-4 w-4" /> New with AI
        </Link>
      </div>

      {isLoading && (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-44 rounded-2xl" />
          ))}
        </div>
      )}

      {isError && (
        <div className="rounded-2xl border border-border bg-card p-8 text-sm text-muted-foreground">
          Couldn't load jobs. Check that the API is reachable.
        </div>
      )}

      {!isLoading && !isError && jobs.length === 0 && (
        <div className="rounded-2xl border border-dashed border-border bg-card p-12 text-center">
          <h3 className="font-display text-xl">No jobs yet.</h3>
          <p className="mt-1 text-sm text-muted-foreground">Create your first role with the AI wizard.</p>
          <Link
            to="/app/jobs/new"
            className="mt-4 inline-flex items-center gap-1.5 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow"
          >
            <Sparkles className="h-4 w-4" /> Create a job
          </Link>
        </div>
      )}

      {!isLoading && !isError && jobs.length > 0 && (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {jobs.map((j: Job, i) => (
            <motion.div
              key={j.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.03 }}
            >
              <Link
                to="/app/jobs/$id"
                params={{ id: j.id }}
                className="group block rounded-2xl border border-border bg-card p-5 transition hover:-translate-y-0.5 hover:shadow-card-soft"
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="text-xs uppercase tracking-widest text-muted-foreground">{j.team}</div>
                    <div className="mt-1 font-display text-xl">{j.title}</div>
                  </div>
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${
                      j.status === "open"
                        ? "bg-success/20 text-success-foreground"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {j.status}
                  </span>
                </div>
                <div className="mt-3 text-sm text-muted-foreground">
                  {j.location} · {j.currency || "$"}
                  {j.salary_min}–{j.salary_max}k
                </div>
                <div className="mt-4 flex flex-wrap gap-1.5">
                  {(j.must_have ?? []).slice(0, 4).map((s) => (
                    <span key={s} className="rounded-full bg-muted px-2 py-0.5 text-[10px]">
                      {s}
                    </span>
                  ))}
                </div>
                <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground">
                  <span>{j.work_mode || j.employment_type || "Full-time"}</span>
                  <span className="text-accent">{j.level || "—"}</span>
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
