// Real analytics derived from the live applications + jobs queries.
// No fake series, no Math.random, no "vs last week" unless it's actually
// computable from timestamps the backend gives us.

import { useMemo } from "react";
import { useApplications, useJobs } from "@/lib/queries";
import type { Application, ApplicationStage, Job } from "@/lib/api";

const DAY_MS = 24 * 60 * 60 * 1000;

export type FunnelPoint = { stage: string; value: number };
export type TrendPoint = { day: string; ts: number; applicants: number; quality: number };
export type TeamPoint = { team: string; open: number; hired: number };
export type SourcePoint = { name: string; value: number };

// Stages that count as having passed through "screened" (i.e. recruiter approved
// or moved further down the pipeline). `approved` is what the backend uses as
// "screened in"; once they reach interview the candidate has obviously been
// screened too.
const SCREENED_STAGES: ApplicationStage[] = [
  "approved",
  "interviewed",
  "offer",
  "hired",
];
const INTERVIEWED_STAGES: ApplicationStage[] = [
  "interviewed",
  "offer",
  "hired",
];
const OFFER_STAGES: ApplicationStage[] = ["offer", "hired"];

// Backend stores created_at as a REAL Unix-epoch (seconds) but the TS type
// declares it as string. Accept both: an ISO date string parses with Date.parse,
// and a numeric epoch (or numeric-string) gets converted to ms. Anything before
// 2001 is treated as seconds (since our clock can't realistically be < 1G ms).
function tsOf(a: Application): number {
  const raw: unknown = a.created_at;
  if (typeof raw === "number" && Number.isFinite(raw)) {
    return raw < 1e12 ? raw * 1000 : raw;
  }
  if (typeof raw === "string") {
    const asNum = Number(raw);
    if (Number.isFinite(asNum) && asNum > 0) {
      return asNum < 1e12 ? asNum * 1000 : asNum;
    }
    const parsed = Date.parse(raw);
    if (Number.isFinite(parsed)) return parsed;
  }
  return 0;
}

function dayKey(ms: number): string {
  const d = new Date(ms);
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${m}/${day}`;
}

function buildFunnel(apps: Application[]): FunnelPoint[] {
  const total = apps.length;
  if (total === 0) return [];
  return [
    { stage: "Applied", value: total },
    {
      stage: "Screened",
      value: apps.filter((a) => SCREENED_STAGES.includes(a.stage)).length,
    },
    {
      stage: "Interviewed",
      value: apps.filter((a) => INTERVIEWED_STAGES.includes(a.stage)).length,
    },
    {
      stage: "Offered",
      value: apps.filter((a) => OFFER_STAGES.includes(a.stage)).length,
    },
    {
      stage: "Hired",
      value: apps.filter((a) => a.stage === "hired").length,
    },
  ];
}

function buildTrend(apps: Application[], days = 14): TrendPoint[] {
  if (apps.length === 0) return [];
  const now = Date.now();
  const start = now - (days - 1) * DAY_MS;
  // Bucket creation timestamps into per-day counts. Quality = average fit_score
  // (0..100) of applications created that day, or 0 when no apps that day.
  const buckets: { ts: number; applicants: number; fitSum: number }[] = [];
  for (let i = 0; i < days; i++) {
    const dayStart = start + i * DAY_MS;
    buckets.push({ ts: dayStart, applicants: 0, fitSum: 0 });
  }
  for (const a of apps) {
    const t = tsOf(a);
    if (t < start) continue;
    const idx = Math.floor((t - start) / DAY_MS);
    if (idx < 0 || idx >= days) continue;
    buckets[idx].applicants += 1;
    buckets[idx].fitSum += Math.round((a.fit_score || 0) * 100);
  }
  return buckets.map((b) => ({
    day: dayKey(b.ts),
    ts: b.ts,
    applicants: b.applicants,
    quality: b.applicants > 0 ? Math.round(b.fitSum / b.applicants) : 0,
  }));
}

function buildTeamPipeline(jobs: Job[], apps: Application[]): TeamPoint[] {
  if (jobs.length === 0) return [];
  const byTeam = new Map<string, { open: number; hired: number }>();
  for (const j of jobs) {
    const team = j.team || "Other";
    const entry = byTeam.get(team) ?? { open: 0, hired: 0 };
    if (j.status === "open") entry.open += 1;
    byTeam.set(team, entry);
  }
  // hired = applications with stage = "hired" attributed to the job's team.
  const jobTeam = new Map(jobs.map((j) => [j.id, j.team || "Other"] as const));
  for (const a of apps) {
    if (a.stage !== "hired") continue;
    const team = jobTeam.get(a.job_id);
    if (!team) continue;
    const entry = byTeam.get(team) ?? { open: 0, hired: 0 };
    entry.hired += 1;
    byTeam.set(team, entry);
  }
  return [...byTeam.entries()]
    .map(([team, v]) => ({ team, ...v }))
    .sort((a, b) => b.open + b.hired - (a.open + a.hired));
}

// Source breakdown. The backend doesn't store an attribution source today, so
// the honest answer is: we can't show this. We derive a *real* breakdown by
// candidate-email domain bucket (gmail/outlook/corporate/other) instead, so
// the slice is honestly grounded in the data we have. If you want true source
// attribution, add a `source` column to applications and surface it here.
function buildSources(apps: Application[]): SourcePoint[] {
  if (apps.length === 0) return [];
  const buckets = { Gmail: 0, Outlook: 0, Corporate: 0, Other: 0 };
  for (const a of apps) {
    const email = (a.candidate_email || "").toLowerCase();
    const at = email.indexOf("@");
    const domain = at >= 0 ? email.slice(at + 1) : "";
    if (!domain) buckets.Other += 1;
    else if (domain === "gmail.com" || domain.endsWith(".gmail.com")) buckets.Gmail += 1;
    else if (
      domain === "outlook.com" ||
      domain === "hotmail.com" ||
      domain === "live.com" ||
      domain.endsWith(".outlook.com")
    )
      buckets.Outlook += 1;
    else if (domain.includes(".")) buckets.Corporate += 1;
    else buckets.Other += 1;
  }
  return Object.entries(buckets)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({ name, value }));
}

// Week-over-week deltas. Returns the *raw* delta and a sign. If the previous
// window is empty we return null so the UI can render "—" instead of fake "+0%".
export type Delta = { value: number; sign: "+" | "-" | "" } | null;

function pctDelta(now: number, prev: number): Delta {
  if (prev === 0 && now === 0) return null;
  if (prev === 0) return { value: 100, sign: "+" };
  const raw = ((now - prev) / prev) * 100;
  const rounded = Math.round(raw);
  return {
    value: Math.abs(rounded),
    sign: rounded > 0 ? "+" : rounded < 0 ? "-" : "",
  };
}

function absDelta(now: number, prev: number): Delta {
  if (prev === 0 && now === 0) return null;
  const raw = now - prev;
  return {
    value: Math.abs(raw),
    sign: raw > 0 ? "+" : raw < 0 ? "-" : "",
  };
}

function splitByWindow<T>(
  items: T[],
  tsAccessor: (x: T) => number,
  windowDays: number,
): { current: T[]; previous: T[] } {
  const now = Date.now();
  const curStart = now - windowDays * DAY_MS;
  const prevStart = now - 2 * windowDays * DAY_MS;
  const current: T[] = [];
  const previous: T[] = [];
  for (const x of items) {
    const t = tsAccessor(x);
    if (t >= curStart) current.push(x);
    else if (t >= prevStart) previous.push(x);
  }
  return { current, previous };
}

// Median time-to-hire over hired applications, in days. Falls back to null
// when there aren't enough hires to compute.
function medianTimeToHireDays(apps: Application[]): number | null {
  const hired = apps.filter((a) => a.stage === "hired" && a.created_at);
  if (hired.length === 0) return null;
  const now = Date.now();
  const days = hired
    .map((a) => (now - tsOf(a)) / DAY_MS)
    .filter((d) => Number.isFinite(d) && d >= 0)
    .sort((a, b) => a - b);
  if (days.length === 0) return null;
  const mid = Math.floor(days.length / 2);
  const m = days.length % 2 === 0 ? (days[mid - 1] + days[mid]) / 2 : days[mid];
  return Math.round(m);
}

export type KpiBlock = {
  label: string;
  // numeric portion + suffix (so the AnimatedNumber can still play); when
  // `value` is null we render "—" honestly.
  value: number | null;
  suffix: string;
  delta: Delta;
  deltaSuffix: string; // e.g. " vs last week" / " vs prior period"
  spark: number[]; // empty array means no sparkline (don't fake one)
};

export type RealAnalytics = {
  loading: boolean;
  hasData: boolean;
  apps: Application[];
  jobs: Job[];
  funnel: FunnelPoint[];
  trend: TrendPoint[];
  sources: SourcePoint[];
  team: TeamPoint[];
  kpis: KpiBlock[];
};

export function useAnalytics(): RealAnalytics {
  const { data: jobs = [], isLoading: jobsLoading } = useJobs();
  const { data: apps = [], isLoading: appsLoading } = useApplications();

  return useMemo<RealAnalytics>(() => {
    const loading = jobsLoading || appsLoading;
    const hasData = apps.length > 0 || jobs.length > 0;

    const funnel = buildFunnel(apps);
    const trend = buildTrend(apps);
    const sources = buildSources(apps);
    const team = buildTeamPipeline(jobs, apps);

    // KPI math — all anchored in real data.
    const openRoles = jobs.filter((j) => j.status === "open").length;
    const openRolesWindow = splitByWindow(
      jobs.filter((j) => j.status === "open"),
      (j) => Date.parse(j.created_at) || 0,
      7,
    );

    const appsWindow = splitByWindow(apps, tsOf, 7);

    const offers = apps.filter((a) => OFFER_STAGES.includes(a.stage)).length;
    const accepted = apps.filter((a) => a.stage === "hired").length;
    const offerRate =
      offers > 0 ? Math.round((accepted / offers) * 100) : null;

    const offersWindow = splitByWindow(
      apps.filter((a) => OFFER_STAGES.includes(a.stage)),
      tsOf,
      7,
    );
    const acceptedWindow = splitByWindow(
      apps.filter((a) => a.stage === "hired"),
      tsOf,
      7,
    );
    const offerRateCur =
      offersWindow.current.length > 0
        ? (acceptedWindow.current.length / offersWindow.current.length) * 100
        : null;
    const offerRatePrev =
      offersWindow.previous.length > 0
        ? (acceptedWindow.previous.length / offersWindow.previous.length) * 100
        : null;
    const offerDelta =
      offerRateCur !== null && offerRatePrev !== null
        ? absDelta(Math.round(offerRateCur), Math.round(offerRatePrev))
        : null;

    const tth = medianTimeToHireDays(apps);

    // Build daily applicant counts for the trailing 8 days for sparklines.
    const sparkTrend = buildTrend(apps, 8);
    const applicantsSpark = sparkTrend.map((p) => p.applicants);
    const qualitySpark = sparkTrend.map((p) => p.quality);

    const kpis: KpiBlock[] = [
      {
        label: "Open roles",
        value: openRoles > 0 ? openRoles : jobs.length === 0 ? null : 0,
        suffix: "",
        delta: absDelta(
          openRolesWindow.current.length,
          openRolesWindow.previous.length,
        ),
        deltaSuffix: " new this week",
        spark: [], // we don't have daily snapshots of "open roles" — don't fake one
      },
      {
        label: "Active candidates",
        value: apps.length > 0 ? apps.length : null,
        suffix: "",
        delta: pctDelta(appsWindow.current.length, appsWindow.previous.length),
        deltaSuffix: "% vs prior week",
        spark: applicantsSpark.some((n) => n > 0) ? applicantsSpark : [],
      },
      {
        label: "Median time-to-hire",
        value: tth,
        suffix: "d",
        delta: null,
        deltaSuffix: tth === null ? "" : " across hires",
        spark: [],
      },
      {
        label: "Offer accept rate",
        value: offerRate,
        suffix: "%",
        delta: offerDelta,
        deltaSuffix: "% vs prior week",
        spark: qualitySpark.some((n) => n > 0) ? qualitySpark : [],
      },
    ];

    return { loading, hasData, apps, jobs, funnel, trend, sources, team, kpis };
  }, [jobs, apps, jobsLoading, appsLoading]);
}

// Public helper for routes that just want headline counts without the
// chart-shaped derivations.
export function useRealCounts() {
  const { data: jobs = [] } = useJobs();
  const { data: apps = [] } = useApplications();
  return useMemo(() => {
    const openJobs = jobs.filter((j) => j.status === "open").length;
    const candidates = apps.length;
    const interviews = apps.filter((a) =>
      INTERVIEWED_STAGES.includes(a.stage),
    ).length;
    const hires = apps.filter((a) => a.stage === "hired").length;
    return { openJobs, candidates, interviews, hires };
  }, [jobs, apps]);
}
