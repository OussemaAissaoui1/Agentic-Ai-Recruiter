import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  ArrowUpRight,
  Briefcase,
  Calendar,
  TrendingUp,
  Users,
  Sparkles,
} from "lucide-react";
import { ANALYTICS, CANDIDATES, JOBS as MOCK_JOBS } from "@/lib/mock";
import { FitBadge, Sparkline, StageChip } from "@/components/Bits";
import { AreaChart, Area, ResponsiveContainer, XAxis, Tooltip } from "recharts";
import { useApplications, useJobs } from "@/lib/queries";
import type { Application } from "@/lib/api";

export const Route = createFileRoute("/app/")({
  head: () => ({ meta: [{ title: "Dashboard — HireFlow" }] }),
  component: Dashboard,
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

function Dashboard() {
  const { data: jobs = [] } = useJobs();
  const { data: apps = [] } = useApplications();

  // Fall back to mocks on a fresh DB so the page still feels alive.
  const useMockApps = apps.length === 0;
  const useMockJobs = jobs.length === 0;

  const openRoles = useMockJobs
    ? MOCK_JOBS.filter((j) => j.status === "Open").length
    : jobs.filter((j) => j.status === "open").length;

  const activeCandidates = useMockApps ? CANDIDATES.length : apps.length;

  const decisionsWaiting = useMockApps
    ? [...CANDIDATES].sort((a, b) => b.fit - a.fit).slice(0, 4).map((c) => ({
        id: c.id,
        name: c.name,
        title: c.title,
        fit: c.fit,
        stage: c.stage,
      }))
    : [...apps]
        .sort((a, b) => b.fit_score - a.fit_score)
        .slice(0, 4)
        .map((a) => ({
          id: a.id,
          name: a.candidate_name || a.candidate_email,
          title: jobs.find((j) => j.id === a.job_id)?.title || "—",
          fit: Math.round(a.fit_score * 100),
          stage: stageToChip(a.stage),
        }));

  const openJobs = useMockJobs
    ? MOCK_JOBS.filter((j) => j.status === "Open").slice(0, 5)
    : jobs.filter((j) => j.status === "open").slice(0, 5);

  const KPIS = [
    {
      label: "Open roles",
      value: String(openRoles),
      delta: "+2",
      icon: Briefcase,
      spark: [4, 6, 5, 7, 8, 9, 11, 12],
    },
    {
      label: "Active candidates",
      value: String(activeCandidates),
      delta: "+38",
      icon: Users,
      spark: [310, 330, 360, 380, 395, 410, 420, 428],
    },
    {
      label: "Time to hire",
      value: "18d",
      delta: "-3d",
      icon: Calendar,
      spark: [25, 24, 22, 21, 20, 19, 19, 18],
    },
    {
      label: "Offer accept rate",
      value: "92%",
      delta: "+4%",
      icon: TrendingUp,
      spark: [78, 80, 84, 86, 88, 90, 91, 92],
    },
  ];

  return (
    <div className="space-y-8">
      <div className="flex items-end justify-between">
        <div>
          <div className="text-xs uppercase tracking-widest text-muted-foreground">Thursday, April 30</div>
          <h1 className="font-display mt-1 text-4xl tracking-tight">Good afternoon, Naomi.</h1>
          <p className="mt-1 text-muted-foreground">
            {useMockApps
              ? "3 candidates need a decision. 2 interviews completed overnight."
              : `${apps.length} applications across ${jobs.length} jobs.`}
          </p>
        </div>
        <Link
          to="/app/jobs/new"
          className="inline-flex items-center gap-1.5 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow"
        >
          <Sparkles className="h-4 w-4" /> New job with AI
        </Link>
      </div>

      <div className="grid gap-3 md:grid-cols-4">
        {KPIS.map((k, i) => (
          <motion.div
            key={k.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className="rounded-2xl border border-border bg-card p-5"
          >
            <div className="flex items-center justify-between">
              <div className="text-xs text-muted-foreground">{k.label}</div>
              <k.icon className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="mt-3 flex items-end justify-between">
              <div>
                <div className="font-display text-3xl">{k.value}</div>
                <div
                  className={`mt-1 text-xs ${
                    k.delta.startsWith("+") ? "text-success-foreground" : "text-destructive"
                  }`}
                >
                  {k.delta} vs last week
                </div>
              </div>
              <div className="text-accent">
                <Sparkline data={k.spark} />
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-2xl border border-border bg-card p-5 lg:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="font-display text-xl">Pipeline velocity</h3>
            <span className="text-xs text-muted-foreground">Last 14 days</span>
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={ANALYTICS.trend}>
                <defs>
                  <linearGradient id="gApp" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="oklch(0.7 0.2 295)" stopOpacity={0.5} />
                    <stop offset="100%" stopColor="oklch(0.7 0.2 295)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gQual" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="oklch(0.72 0.16 155)" stopOpacity={0.5} />
                    <stop offset="100%" stopColor="oklch(0.72 0.16 155)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="day"
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 10, fill: "currentColor", opacity: 0.5 }}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--color-popover)",
                    border: "1px solid var(--color-border)",
                    borderRadius: 12,
                    fontSize: 12,
                  }}
                />
                <Area type="monotone" dataKey="applicants" stroke="oklch(0.7 0.2 295)" strokeWidth={2} fill="url(#gApp)" />
                <Area type="monotone" dataKey="quality" stroke="oklch(0.72 0.16 155)" strokeWidth={2} fill="url(#gQual)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-card p-5">
          <h3 className="font-display text-xl">Decisions waiting</h3>
          <div className="mt-4 space-y-3">
            {decisionsWaiting.map((c) => (
              <Link
                key={c.id}
                to="/app/applicants"
                search={{ c: c.id }}
                className="flex items-center gap-3 rounded-xl p-2 -mx-2 transition hover:bg-muted"
              >
                <FitBadge value={c.fit} size="sm" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium">{c.name}</div>
                  <div className="truncate text-xs text-muted-foreground">{c.title}</div>
                </div>
                <StageChip stage={c.stage} />
              </Link>
            ))}
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-card">
        <div className="flex items-center justify-between border-b border-border p-5">
          <h3 className="font-display text-xl">Open roles</h3>
          <Link
            to="/app/jobs"
            className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
          >
            View all <ArrowUpRight className="h-3.5 w-3.5" />
          </Link>
        </div>
        <div className="divide-y divide-border">
          {openJobs.map((j) => {
            const id = "id" in j ? j.id : "";
            const title = j.title;
            const team = j.team;
            const location = j.location;
            const applicantsCount = useMockJobs
              ? (j as (typeof MOCK_JOBS)[number]).applicants
              : apps.filter((a) => a.job_id === id).length;
            const newToday = useMockJobs
              ? (j as (typeof MOCK_JOBS)[number]).newToday
              : apps.filter(
                  (a) =>
                    a.job_id === id &&
                    Date.now() - new Date(a.created_at).getTime() < 24 * 60 * 60 * 1000,
                ).length;
            return (
              <Link
                key={id}
                to="/app/jobs/$id"
                params={{ id }}
                className="flex items-center gap-4 p-4 transition hover:bg-muted/40"
              >
                <div className="grid h-10 w-10 place-items-center rounded-xl bg-muted">
                  <Briefcase className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-semibold">{title}</div>
                  <div className="truncate text-xs text-muted-foreground">
                    {team} · {location}
                  </div>
                </div>
                <div className="hidden text-right md:block">
                  <div className="text-xs text-muted-foreground">Applicants</div>
                  <div className="font-mono text-sm">{applicantsCount}</div>
                </div>
                <div className="hidden text-right md:block">
                  <div className="text-xs text-muted-foreground">New today</div>
                  <div className="font-mono text-sm text-accent">+{newToday}</div>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}
