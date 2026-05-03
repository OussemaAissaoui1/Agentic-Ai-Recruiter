import { createFileRoute, Link } from "@tanstack/react-router";
import { FitBadge, StageChip } from "@/components/Bits";
import {
  ArrowLeft,
  MapPin,
  Briefcase as BIcon,
  DollarSign,
  Calendar,
} from "lucide-react";
import { useApplications, useJob } from "@/lib/queries";
import { Skeleton } from "@/components/ui/skeleton";
import type { Application } from "@/lib/api";

export const Route = createFileRoute("/app/jobs/$id")({
  head: ({ params }) => ({ meta: [{ title: `${params.id} — HireFlow` }] }),
  component: JobDetail,
  notFoundComponent: () => <div>Job not found</div>,
});

function stageToChip(s: Application["stage"]): "New" | "Interview" | "Offer" | "Rejected" {
  switch (s) {
    case "applied":
      return "New";
    case "approved":
    case "interviewed":
      return "Interview";
    case "offer":
    case "hired":
      return "Offer";
    case "rejected":
      return "Rejected";
    default:
      return "New";
  }
}

function JobDetail() {
  const { id } = Route.useParams();
  const { data: job, isLoading, isError } = useJob(id);
  const { data: apps = [] } = useApplications({ job_id: id });

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-6 w-32" />
        <Skeleton className="h-12 w-3/4" />
        <Skeleton className="h-72 w-full rounded-2xl" />
      </div>
    );
  }
  if (isError || !job) return <div>Job not found</div>;

  const sorted = [...apps].sort((a, b) => b.fit_score - a.fit_score);
  const newToday = apps.filter((a) => {
    const t = new Date(a.created_at).getTime();
    return Date.now() - t < 24 * 60 * 60 * 1000;
  }).length;

  return (
    <div className="space-y-8">
      <Link
        to="/app/jobs"
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" /> All jobs
      </Link>
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <div>
            <div className="text-xs uppercase tracking-widest text-muted-foreground">{job.team}</div>
            <h1 className="font-display mt-1 text-4xl tracking-tight">{job.title}</h1>
            <div className="mt-4 flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
              <span className="inline-flex items-center gap-1">
                <MapPin className="h-4 w-4" />
                {job.location}
              </span>
              <span className="inline-flex items-center gap-1">
                <BIcon className="h-4 w-4" />
                {job.employment_type || "Full-time"}
              </span>
              <span className="inline-flex items-center gap-1">
                <DollarSign className="h-4 w-4" />
                {job.currency || "$"}
                {job.salary_min}–{job.salary_max}k
              </span>
              <span className="inline-flex items-center gap-1">
                <Calendar className="h-4 w-4" />
                Posted {new Date(job.created_at).toLocaleDateString()}
              </span>
            </div>
          </div>
          <div className="rounded-2xl border border-border bg-card p-6">
            <h3 className="font-display text-xl">Description</h3>
            <p className="mt-3 whitespace-pre-wrap text-pretty text-muted-foreground">{job.description}</p>
            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              <div>
                <div className="text-xs uppercase tracking-widest text-muted-foreground">Must-haves</div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {(job.must_have ?? []).map((s) => (
                    <span
                      key={s}
                      className="rounded-full bg-accent/15 px-2.5 py-1 text-xs text-accent"
                    >
                      {s}
                    </span>
                  ))}
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-widest text-muted-foreground">Nice-to-haves</div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {(job.nice_to_have ?? []).map((s) => (
                    <span key={s} className="rounded-full bg-muted px-2.5 py-1 text-xs">
                      {s}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
        <aside className="space-y-3">
          <div className="rounded-2xl border border-border bg-card p-5">
            <div className="text-xs uppercase tracking-widest text-muted-foreground">Pipeline health</div>
            <div className="mt-2 font-display text-3xl">{apps.length}</div>
            <div className="text-sm text-accent">+{newToday} new today</div>
          </div>
          <div className="rounded-2xl border border-border bg-card p-5">
            <div className="mb-3 text-sm font-semibold">Top candidates</div>
            <div className="space-y-3">
              {sorted.length === 0 && (
                <div className="text-xs text-muted-foreground">No applications yet.</div>
              )}
              {sorted.slice(0, 5).map((c) => (
                <Link
                  key={c.id}
                  to="/app/applicants"
                  search={{ c: c.id }}
                  className="flex items-center gap-3 rounded-lg p-1 -m-1 hover:bg-muted"
                >
                  <FitBadge value={Math.round(c.fit_score * 100)} size="sm" />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm">{c.candidate_name || c.candidate_email}</div>
                    <StageChip stage={stageToChip(c.stage)} />
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
