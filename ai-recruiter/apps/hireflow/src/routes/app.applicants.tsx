import { createFileRoute, useSearch } from "@tanstack/react-router";
import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FitBadge, StageChip } from "@/components/Bits";
import {
  Filter,
  GitCompare,
  X,
  ChevronRight,
  MapPin,
  Briefcase,
  Calendar,
  Loader2,
} from "lucide-react";
import {
  useApplications,
  useInviteInterview,
  useJobs,
  useUpdateApplication,
} from "@/lib/queries";
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "sonner";
import type { Application, Job } from "@/lib/api";

export const Route = createFileRoute("/app/applicants")({
  head: () => ({ meta: [{ title: "Applicants — HireFlow" }] }),
  validateSearch: (s: Record<string, unknown>) => ({
    c: typeof s.c === "string" ? s.c : undefined,
  }),
  component: Applicants,
});

type ChipStage = "New" | "Interview" | "Offer" | "Rejected";

function stageToChip(s: Application["stage"]): ChipStage {
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

function initialsOf(name: string, email: string): string {
  const src = (name || email || "??").trim();
  const parts = src.split(/\s+|@|\./).filter(Boolean);
  if (parts.length === 0) return "??";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[1][0]).toUpperCase();
}

function avatarHueOf(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h % 360;
}

function Applicants() {
  const search = useSearch({ from: "/app/applicants" });
  const [jobFilter, setJobFilter] = useState<string>("all");
  const [stage, setStage] = useState<string>("all");
  const [minFit, setMinFit] = useState(0);
  const [selected, setSelected] = useState<string | null>(search.c ?? null);
  const [compare, setCompare] = useState<string[]>([]);

  const { data: jobs = [] } = useJobs();
  const { data: apps = [], isLoading } = useApplications();

  const filtered = useMemo(() => {
    return apps
      .filter(
        (c) =>
          (jobFilter === "all" || c.job_id === jobFilter) &&
          (stage === "all" || stageToChip(c.stage) === stage) &&
          Math.round(c.fit_score * 100) >= minFit,
      )
      .sort((a, b) => b.fit_score - a.fit_score);
  }, [apps, jobFilter, stage, minFit]);

  const sel = apps.find((c) => c.id === selected);
  const toggleCompare = (id: string) => {
    setCompare((prev) =>
      prev.includes(id)
        ? prev.filter((x) => x !== id)
        : prev.length < 3
          ? [...prev, id]
          : prev,
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="font-display text-4xl tracking-tight">Applicants</h1>
          <p className="mt-1 text-muted-foreground">
            {isLoading
              ? "Loading…"
              : `${filtered.length} candidates · ranked by AI fit score`}
          </p>
        </div>
        {compare.length >= 2 && (
          <button
            onClick={() => setSelected("__compare__")}
            className="inline-flex items-center gap-1.5 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow"
          >
            <GitCompare className="h-4 w-4" /> Compare {compare.length}
          </button>
        )}
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-2 rounded-2xl border border-border bg-card p-3">
        <Filter className="h-4 w-4 text-muted-foreground" />
        <select
          value={jobFilter}
          onChange={(e) => setJobFilter(e.target.value)}
          className="rounded-lg border border-border bg-background px-3 py-1.5 text-sm"
        >
          <option value="all">All jobs</option>
          {jobs.map((j) => (
            <option key={j.id} value={j.id}>
              {j.title}
            </option>
          ))}
        </select>
        <select
          value={stage}
          onChange={(e) => setStage(e.target.value)}
          className="rounded-lg border border-border bg-background px-3 py-1.5 text-sm"
        >
          {["all", "New", "Interview", "Offer", "Rejected"].map((s) => (
            <option key={s} value={s}>
              {s === "all" ? "All stages" : s}
            </option>
          ))}
        </select>
        <div className="ml-2 flex items-center gap-2 text-sm text-muted-foreground">
          Min fit
          <input
            type="range"
            min={0}
            max={100}
            value={minFit}
            onChange={(e) => setMinFit(+e.target.value)}
            className="w-32 accent-[oklch(0.7_0.2_295)]"
          />
          <span className="font-mono w-8">{minFit}</span>
        </div>
      </div>

      {isLoading && (
        <div className="space-y-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full rounded-xl" />
          ))}
        </div>
      )}

      {!isLoading && filtered.length === 0 && (
        <div className="rounded-2xl border border-dashed border-border bg-card p-12 text-center text-sm text-muted-foreground">
          No applicants match these filters yet.
        </div>
      )}

      {!isLoading && filtered.length > 0 && (
        <div className="overflow-hidden rounded-2xl border border-border bg-card">
          <div className="grid grid-cols-12 gap-3 border-b border-border bg-muted/30 px-4 py-3 text-xs uppercase tracking-widest text-muted-foreground">
            <div className="col-span-1"></div>
            <div className="col-span-4">Candidate</div>
            <div className="col-span-2">Role</div>
            <div className="col-span-2 hidden md:block">Skills</div>
            <div className="col-span-1 text-center">Stage</div>
            <div className="col-span-2 text-right">Fit</div>
          </div>
          {filtered.slice(0, 30).map((c, i) => {
            const fit = Math.round(c.fit_score * 100);
            const job = jobs.find((j) => j.id === c.job_id);
            const initials = initialsOf(c.candidate_name, c.candidate_email);
            const hue = avatarHueOf(c.candidate_email || c.id);
            return (
              <motion.button
                key={c.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: Math.min(i * 0.015, 0.4) }}
                onClick={() => setSelected(c.id)}
                className="grid w-full grid-cols-12 items-center gap-3 border-b border-border px-4 py-3 text-left transition hover:bg-muted/40"
              >
                <div className="col-span-1">
                  <input
                    type="checkbox"
                    checked={compare.includes(c.id)}
                    onChange={(e) => {
                      e.stopPropagation();
                      toggleCompare(c.id);
                    }}
                    onClick={(e) => e.stopPropagation()}
                    className="h-4 w-4 accent-[oklch(0.7_0.2_295)]"
                  />
                </div>
                <div className="col-span-4 flex items-center gap-3">
                  <div
                    className="grid h-9 w-9 place-items-center rounded-full text-xs font-semibold text-white"
                    style={{
                      background: `linear-gradient(135deg, oklch(0.7 0.18 ${hue}), oklch(0.6 0.2 ${(hue + 60) % 360}))`,
                    }}
                  >
                    {initials}
                  </div>
                  <div className="min-w-0">
                    <div className="truncate font-medium">
                      {c.candidate_name || c.candidate_email || "Unknown"}
                    </div>
                    <div className="truncate text-xs text-muted-foreground">
                      {c.candidate_email}
                    </div>
                  </div>
                </div>
                <div className="col-span-2 truncate text-sm text-muted-foreground">
                  {job?.title || "—"}
                </div>
                <div className="col-span-2 hidden flex-wrap gap-1 md:flex">
                  {(c.matched_skills ?? []).slice(0, 3).map((s) => (
                    <span key={s} className="rounded-full bg-accent/10 px-1.5 py-0.5 text-[10px] text-accent">
                      {s}
                    </span>
                  ))}
                </div>
                <div className="col-span-1 text-center">
                  <StageChip stage={stageToChip(c.stage)} />
                </div>
                <div className="col-span-2 flex items-center justify-end gap-2">
                  <FitBadge value={fit} size="sm" />
                  <ChevronRight className="h-4 w-4 text-muted-foreground" />
                </div>
              </motion.button>
            );
          })}
        </div>
      )}

      <AnimatePresence>
        {sel && selected !== "__compare__" && (
          <Drawer onClose={() => setSelected(null)}>
            <CandidatePanel c={sel} job={jobs.find((j) => j.id === sel.job_id)} />
          </Drawer>
        )}
        {selected === "__compare__" && (
          <Drawer onClose={() => setSelected(null)} wide>
            <ComparePanel apps={compare.map((id) => apps.find((a) => a.id === id)).filter(Boolean) as Application[]} />
          </Drawer>
        )}
      </AnimatePresence>
    </div>
  );
}

function Drawer({
  children,
  onClose,
  wide,
}: {
  children: React.ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  return (
    <>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="fixed inset-0 z-40 bg-foreground/20 backdrop-blur-sm"
      />
      <motion.aside
        initial={{ x: "100%" }}
        animate={{ x: 0 }}
        exit={{ x: "100%" }}
        transition={{ type: "spring", stiffness: 320, damping: 36 }}
        className={`fixed right-0 top-0 z-50 h-screen w-full overflow-y-auto border-l border-border bg-card shadow-2xl ${
          wide ? "max-w-5xl" : "max-w-xl"
        }`}
      >
        <button
          onClick={onClose}
          className="sticky top-3 ml-auto mr-3 flex rounded-md p-1.5 hover:bg-muted"
        >
          <X className="h-4 w-4" />
        </button>
        <div className="px-6 pb-10 -mt-7">{children}</div>
      </motion.aside>
    </>
  );
}

function CandidatePanel({ c, job }: { c: Application; job?: Job }) {
  const fit = Math.round(c.fit_score * 100);
  const hue = avatarHueOf(c.candidate_email || c.id);
  const initials = initialsOf(c.candidate_name, c.candidate_email);
  const update = useUpdateApplication();
  const invite = useInviteInterview();

  const moveTo = async (next: Application["stage"]) => {
    try {
      await update.mutateAsync({ id: c.id, body: { stage: next } });
      toast.success(`Moved to ${next}.`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  };

  const sendInvite = async () => {
    try {
      await invite.mutateAsync(c.id);
      toast.success("Interview invite sent.");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <div
          className="grid h-16 w-16 place-items-center rounded-2xl text-lg font-semibold text-white"
          style={{
            background: `linear-gradient(135deg, oklch(0.7 0.18 ${hue}), oklch(0.6 0.2 ${(hue + 60) % 360}))`,
          }}
        >
          {initials}
        </div>
        <div className="min-w-0 flex-1">
          <h2 className="font-display text-2xl">
            {c.candidate_name || c.candidate_email || "Unknown"}
          </h2>
          <div className="text-sm text-muted-foreground">{c.candidate_email}</div>
          <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-1">
              <Briefcase className="h-3 w-3" />
              {job?.title || "—"}
            </span>
            {job?.location && (
              <span className="inline-flex items-center gap-1">
                <MapPin className="h-3 w-3" />
                {job.location}
              </span>
            )}
            <span className="inline-flex items-center gap-1">
              <Calendar className="h-3 w-3" />
              {new Date(c.created_at).toLocaleDateString()}
            </span>
          </div>
        </div>
        <FitBadge value={fit} size="lg" />
      </div>

      <div className="rounded-2xl border border-border bg-background p-4">
        <div className="flex items-center justify-between gap-2">
          <div className="text-xs uppercase tracking-widest text-muted-foreground">CV</div>
          <span className="truncate text-xs text-muted-foreground">
            {c.cv_filename || "—"}
          </span>
        </div>
        {c.cv_text ? (
          <details className="mt-3 group">
            <summary className="cursor-pointer text-sm text-accent hover:underline">
              <span className="group-open:hidden">Read full CV</span>
              <span className="hidden group-open:inline">Hide CV</span>
            </summary>
            <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap rounded-lg border border-border bg-muted/30 p-3 font-mono text-xs leading-relaxed">
              {c.cv_text}
            </pre>
          </details>
        ) : (
          <p className="mt-2 text-xs italic text-muted-foreground">
            No parsed text available for this CV.
          </p>
        )}
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-2xl border border-border p-4">
          <div className="text-xs uppercase tracking-widest text-muted-foreground">Matched</div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {(c.matched_skills ?? []).map((s) => (
              <span key={s} className="rounded-full bg-success/20 px-2 py-0.5 text-xs">
                {s}
              </span>
            ))}
          </div>
        </div>
        <div className="rounded-2xl border border-border p-4">
          <div className="text-xs uppercase tracking-widest text-muted-foreground">Gaps</div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {(c.missing_skills ?? []).length ? (
              (c.missing_skills ?? []).map((s) => (
                <span key={s} className="rounded-full bg-destructive/15 px-2 py-0.5 text-xs">
                  {s}
                </span>
              ))
            ) : (
              <span className="text-xs text-muted-foreground">None critical</span>
            )}
          </div>
        </div>
      </div>

      {c.notes && (
        <div className="rounded-2xl border border-border bg-background p-4">
          <div className="text-xs uppercase tracking-widest text-muted-foreground">Notes</div>
          <p className="mt-2 text-sm">{c.notes}</p>
        </div>
      )}

      <div className="text-xs text-muted-foreground">
        Applied for <span className="font-medium text-foreground">{job?.title}</span>
      </div>

      <div className="flex flex-wrap gap-2 border-t border-border pt-4">
        <button
          disabled={invite.isPending || c.stage === "approved" || c.stage === "interviewed"}
          onClick={sendInvite}
          className="inline-flex items-center gap-2 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow disabled:opacity-60"
        >
          {invite.isPending && <Loader2 className="h-4 w-4 animate-spin" />}
          {c.stage === "approved" || c.stage === "interviewed"
            ? "Approved for interview"
            : "Approve for interview"}
        </button>
        <button
          disabled={update.isPending || c.stage === "rejected"}
          onClick={() => moveTo("rejected")}
          className="rounded-full px-4 py-2 text-sm text-muted-foreground hover:text-destructive disabled:opacity-60"
        >
          Reject
        </button>
      </div>
    </div>
  );
}

function ComparePanel({ apps }: { apps: Application[] }) {
  return (
    <div className="space-y-6">
      <h2 className="font-display text-2xl">Side-by-side</h2>
      <div
        className="grid gap-4"
        style={{ gridTemplateColumns: `repeat(${Math.max(apps.length, 1)}, minmax(0, 1fr))` }}
      >
        {apps.map((c) => {
          const fit = Math.round(c.fit_score * 100);
          const hue = avatarHueOf(c.candidate_email || c.id);
          const initials = initialsOf(c.candidate_name, c.candidate_email);
          return (
            <div key={c.id} className="rounded-2xl border border-border bg-background p-4">
              <div className="flex items-center gap-3">
                <div
                  className="grid h-10 w-10 place-items-center rounded-full text-xs font-semibold text-white"
                  style={{
                    background: `linear-gradient(135deg, oklch(0.7 0.18 ${hue}), oklch(0.6 0.2 ${(hue + 60) % 360}))`,
                  }}
                >
                  {initials}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-semibold">
                    {c.candidate_name || c.candidate_email}
                  </div>
                  <div className="truncate text-xs text-muted-foreground">{c.candidate_email}</div>
                </div>
                <FitBadge value={fit} size="sm" />
              </div>
              <div className="mt-3 text-xs text-muted-foreground">
                Stage: <StageChip stage={stageToChip(c.stage)} />
              </div>
              <div className="mt-2 flex flex-wrap gap-1">
                {(c.matched_skills ?? []).slice(0, 6).map((s) => (
                  <span key={s} className="rounded-full bg-accent/10 px-2 py-0.5 text-[10px] text-accent">
                    {s}
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
