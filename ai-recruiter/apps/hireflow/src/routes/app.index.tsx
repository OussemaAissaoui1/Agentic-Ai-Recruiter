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
import { FitBadge, Sparkline, StageChip } from "@/components/Bits";
import { AnimatedNumber } from "@/components/motion/AnimatedNumber";
import { SpotlightCard } from "@/components/motion/SpotlightCard";
import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis, Tooltip } from "recharts";
import { useAnalytics } from "@/lib/analytics";
import type { Application } from "@/lib/api";

function createdAtMs(a: Application): number {
  const raw: unknown = a.created_at;
  if (typeof raw === "number" && Number.isFinite(raw)) {
    return raw < 1e12 ? raw * 1000 : raw;
  }
  if (typeof raw === "string") {
    const n = Number(raw);
    if (Number.isFinite(n) && n > 0) return n < 1e12 ? n * 1000 : n;
    const p = Date.parse(raw);
    if (Number.isFinite(p)) return p;
  }
  return 0;
}

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

function todayLabel() {
  return new Date().toLocaleDateString(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
  });
}

function greetingFor(d = new Date()) {
  const h = d.getHours();
  if (h < 5) return "Working late";
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

function Dashboard() {
  const { apps, jobs, kpis, trend, loading } = useAnalytics();

  const openJobs = jobs.filter((j) => j.status === "open").slice(0, 5);

  const decisionsWaiting = [...apps]
    .filter((a) => a.stage === "applied")
    .sort((a, b) => b.fit_score - a.fit_score)
    .slice(0, 4)
    .map((a) => ({
      id: a.id,
      name: a.candidate_name || a.candidate_email,
      title: jobs.find((j) => j.id === a.job_id)?.title || "—",
      fit: Math.round(a.fit_score * 100),
      stage: stageToChip(a.stage),
    }));

  const waitingCount = apps.filter((a) => a.stage === "applied").length;
  const interviewedCount = apps.filter((a) => a.stage === "interviewed").length;
  const hasTrend = trend.some((p) => p.applicants > 0);

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="text-xs uppercase tracking-widest text-muted-foreground">
            {todayLabel()}
          </div>
          <h1 className="font-display mt-1 text-3xl tracking-tight">
            {greetingFor()}.
          </h1>
          <p className="mt-1 text-muted-foreground">
            {loading
              ? "Loading your pipeline…"
              : apps.length === 0
                ? "No applications yet. Post a role to start receiving candidates."
                : `${waitingCount} ${waitingCount === 1 ? "candidate is" : "candidates are"} waiting on you · ${interviewedCount} interviewed.`}
          </p>
        </div>
        <Link
          to="/app/explorer"
          className="inline-flex min-h-[44px] items-center gap-1.5 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow"
        >
          <Sparkles className="h-4 w-4" /> New job with AI
        </Link>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-4">
        {kpis.map((k, i) => {
          const Icon = [Briefcase, Users, Calendar, TrendingUp][i] ?? Briefcase;
          const isPositive = k.delta?.sign === "+";
          const deltaColor =
            !k.delta || k.delta.sign === ""
              ? "text-muted-foreground"
              : isPositive
                ? "text-success-foreground"
                : "text-destructive";
          return (
            <motion.div
              key={k.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
            >
              <SpotlightCard className="lift h-full rounded-2xl border border-border bg-card p-5">
                <div className="flex items-center justify-between">
                  <div className="text-xs text-muted-foreground">{k.label}</div>
                  <Icon className="h-4 w-4 text-muted-foreground" aria-hidden />
                </div>
                <div className="mt-3 flex items-end justify-between gap-3">
                  <div className="min-w-0">
                    <div className="font-display text-3xl tabular-nums">
                      {k.value === null ? (
                        <span className="text-muted-foreground">—</span>
                      ) : (
                        <>
                          <AnimatedNumber value={k.value} duration={1100} />
                          {k.suffix}
                        </>
                      )}
                    </div>
                    <div className={`mt-1 text-xs ${deltaColor}`}>
                      {k.delta
                        ? `${k.delta.sign}${k.delta.value}${k.deltaSuffix}`
                        : k.value === null
                          ? "No data yet"
                          : "No prior period"}
                    </div>
                  </div>
                  {k.spark.length > 1 && (
                    <div className="text-accent" aria-hidden>
                      <Sparkline data={k.spark} />
                    </div>
                  )}
                </div>
              </SpotlightCard>
            </motion.div>
          );
        })}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-2xl border border-border bg-card p-5 lg:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="font-display text-xl">Pipeline velocity</h3>
            <span className="text-xs text-muted-foreground">Last 14 days</span>
          </div>
          <div className="h-56">
            {hasTrend ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trend}>
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
                  <YAxis hide />
                  <Tooltip
                    contentStyle={{
                      background: "var(--color-popover)",
                      border: "1px solid var(--color-border)",
                      borderRadius: 12,
                      fontSize: 12,
                    }}
                    formatter={(v: number, key) =>
                      key === "quality" ? [`${v} fit`, "Avg quality"] : [v, "Applicants"]
                    }
                  />
                  <Area
                    type="monotone"
                    dataKey="applicants"
                    name="Applicants"
                    stroke="oklch(0.7 0.2 295)"
                    strokeWidth={2}
                    fill="url(#gApp)"
                  />
                  <Area
                    type="monotone"
                    dataKey="quality"
                    name="Avg fit"
                    stroke="oklch(0.72 0.16 155)"
                    strokeWidth={2}
                    fill="url(#gQual)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState
                title="No applications in the last 14 days"
                hint="When candidates apply this chart will fill in automatically."
              />
            )}
          </div>
        </div>

        <div className="rounded-2xl border border-border bg-card p-5">
          <h3 className="font-display text-xl">Decisions waiting</h3>
          {decisionsWaiting.length === 0 ? (
            <div className="mt-4">
              <EmptyState
                title="You're all caught up"
                hint="Every applicant has a decision. New applications will appear here."
              />
            </div>
          ) : (
            <div className="mt-4 space-y-3">
              {decisionsWaiting.map((c) => (
                <Link
                  key={c.id}
                  to="/app/applicants"
                  search={{ c: c.id }}
                  className="flex min-h-[44px] items-center gap-3 rounded-xl p-2 -mx-2 transition hover:bg-muted focus-visible:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
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
          )}
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-card">
        <div className="flex items-center justify-between border-b border-border p-5">
          <h3 className="font-display text-xl">Open roles</h3>
          <Link
            to="/app/jobs"
            className="inline-flex min-h-[36px] items-center gap-1 rounded-md px-2 text-sm text-muted-foreground transition hover:text-foreground focus-visible:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            View all <ArrowUpRight className="h-3.5 w-3.5" />
          </Link>
        </div>
        {openJobs.length === 0 ? (
          <div className="p-8">
            <EmptyState
              title="No open roles yet"
              hint="Create a role to start collecting applications."
              action={
                <Link
                  to="/app/explorer"
                  className="inline-flex min-h-[40px] items-center gap-1.5 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow"
                >
                  <Sparkles className="h-4 w-4" /> New job with AI
                </Link>
              }
            />
          </div>
        ) : (
          <div className="divide-y divide-border">
            {openJobs.map((j) => {
              const id = j.id;
              const applicantsCount = apps.filter((a) => a.job_id === id).length;
              const newToday = apps.filter(
                (a) =>
                  a.job_id === id &&
                  Date.now() - createdAtMs(a) < 24 * 60 * 60 * 1000,
              ).length;
              return (
                <Link
                  key={id}
                  to="/app/jobs/$id"
                  params={{ id }}
                  className="flex min-h-[64px] items-center gap-4 p-4 transition hover:bg-muted/40 focus-visible:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                >
                  <div className="grid h-10 w-10 place-items-center rounded-xl bg-muted">
                    <Briefcase className="h-4 w-4" aria-hidden />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-semibold">{j.title}</div>
                    <div className="truncate text-xs text-muted-foreground">
                      {j.team} · {j.location}
                    </div>
                  </div>
                  <div className="hidden text-right md:block">
                    <div className="text-xs text-muted-foreground">Applicants</div>
                    <div className="font-mono text-sm tabular-nums">{applicantsCount}</div>
                  </div>
                  <div className="hidden text-right md:block">
                    <div className="text-xs text-muted-foreground">New today</div>
                    <div
                      className={`font-mono text-sm tabular-nums ${newToday > 0 ? "text-accent" : "text-muted-foreground"}`}
                    >
                      {newToday > 0 ? `+${newToday}` : "0"}
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function EmptyState({
  title,
  hint,
  action,
}: {
  title: string;
  hint?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 rounded-xl bg-muted/30 p-6 text-center">
      <div className="text-sm font-medium">{title}</div>
      {hint && <div className="text-xs text-muted-foreground">{hint}</div>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}
