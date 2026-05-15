import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  ArrowRight,
  Briefcase,
  Building2,
  CircleCheck,
  Clock,
  MapPin,
  Sparkles,
  Wallet,
} from "lucide-react";
import { useApplications, useJob } from "@/lib/queries";
import { useApp } from "@/lib/store";
import { Skeleton } from "@/components/ui/skeleton";
import { SpotlightCard } from "@/components/motion/SpotlightCard";
import type { Application, ApplicationStage } from "@/lib/api";

export const Route = createFileRoute("/c/jobs/$id")({
  head: ({ params }) => ({
    meta: [{ title: `Role ${params.id} — HireFlow` }],
  }),
  component: JobDetail,
});

function JobDetail() {
  const { id } = Route.useParams();
  const { data: job, isLoading, isError } = useJob(id);
  const { candidateEmail } = useApp();
  const { data: existingApps = [] } = useApplications(
    candidateEmail
      ? { job_id: id, candidate_email: candidateEmail }
      : { job_id: id },
  );
  const myApplication = candidateEmail
    ? existingApps.find(
        (a) =>
          a.candidate_email?.toLowerCase() === candidateEmail.toLowerCase(),
      )
    : undefined;

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-6 w-32" />
        <Skeleton className="h-40 w-full rounded-2xl" />
        <Skeleton className="h-72 w-full rounded-2xl" />
      </div>
    );
  }

  if (isError || !job) {
    return (
      <div className="space-y-6">
        <Link
          to="/c"
          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          All roles
        </Link>
        <div className="rounded-2xl border border-dashed border-border bg-card p-12 text-center text-sm text-muted-foreground">
          This role isn't available anymore.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <Link
        to="/c"
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" />
        All roles
      </Link>

      {/* Hero */}
      <SpotlightCard
        as="section"
        className="rounded-2xl border border-border bg-card p-6 md:p-8 motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-2 motion-safe:duration-500"
      >
        <div className="flex flex-col gap-6 md:flex-row md:items-start md:justify-between">
          <div className="min-w-0">
            <div className="inline-flex items-center gap-2 text-xs uppercase tracking-widest text-muted-foreground">
              <Building2 className="h-3.5 w-3.5" />
              {job.team || "Hiring team"}
            </div>
            <h1 className="font-display mt-2 text-3xl tracking-tight md:text-4xl">
              {job.title}
            </h1>
            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-muted-foreground">
              {job.location && (
                <span className="inline-flex items-center gap-1">
                  <MapPin className="h-3.5 w-3.5" />
                  {job.location}
                </span>
              )}
              {job.work_mode && (
                <span className="inline-flex items-center gap-1">
                  <Briefcase className="h-3.5 w-3.5" />
                  {prettify(job.work_mode)}
                </span>
              )}
              {job.employment_type && (
                <span className="inline-flex items-center gap-1">
                  <Clock className="h-3.5 w-3.5" />
                  {prettify(job.employment_type)}
                </span>
              )}
              {(job.salary_min || job.salary_max) && (
                <span className="inline-flex items-center gap-1">
                  <Wallet className="h-3.5 w-3.5" />
                  {job.currency || "$"}
                  {job.salary_min}–{job.salary_max}k
                </span>
              )}
            </div>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {job.level && <Chip>{prettify(job.level)}</Chip>}
              <Chip
                tone={job.status === "open" ? "success" : "muted"}
              >
                {job.status === "open" ? "Open" : "Closed"}
              </Chip>
            </div>
          </div>

          <div className="w-full shrink-0 md:w-auto md:text-right">
            {myApplication ? (
              <AppliedFitCard application={myApplication} />
            ) : (
              <div className="rounded-2xl border border-dashed border-accent/40 bg-accent/10 p-4 md:max-w-xs md:text-left">
                <div className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-accent">
                  <Sparkles className="h-3 w-3" />
                  Your fit — locked
                </div>
                <div className="mt-2 font-display text-xl leading-tight tracking-tight text-foreground">
                  See how well you match
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Submit your CV and we'll compute your personal fit score
                  with full evidence — skills you match, gaps to address, the
                  lines from your CV that earned the points.
                </p>
              </div>
            )}
            {myApplication ? (
              <AppliedHeroCta application={myApplication} />
            ) : (
              <Link
                to="/c/apply/$id"
                params={{ id: job.id }}
                aria-disabled={job.status !== "open"}
                className={`group mt-4 inline-flex items-center gap-1.5 rounded-full px-4 py-2 text-sm font-semibold shadow-glow press-tight transition-transform ${
                  job.status === "open"
                    ? "bg-violet-grad text-accent-foreground hover:scale-[1.03] glow-pulse"
                    : "pointer-events-none cursor-not-allowed bg-muted text-muted-foreground"
                }`}
              >
                {job.status === "open" ? "Apply to see your fit" : "Closed"}
                {job.status === "open" && (
                  <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
                )}
              </Link>
            )}
          </div>
        </div>
      </SpotlightCard>

      {/* Description */}
      <motion.section
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="rounded-2xl border border-border bg-card p-6 md:p-8"
      >
        <h2 className="font-display text-xl tracking-tight">About this role</h2>
        {job.description ? (
          <div className="prose prose-sm mt-4 max-w-none whitespace-pre-wrap text-pretty text-foreground/90 dark:prose-invert">
            {job.description}
          </div>
        ) : (
          <p className="mt-4 text-sm italic text-muted-foreground">
            No description provided yet for this role.
          </p>
        )}
      </motion.section>

      {/* Skills */}
      <motion.section
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="grid gap-4 md:grid-cols-2"
      >
        <SkillBlock
          heading="Must-have skills"
          subtitle="What the team expects you to bring on day one."
          skills={job.must_have ?? []}
          tone="accent"
        />
        <SkillBlock
          heading="Nice-to-have skills"
          subtitle="Bonus signals — not a hard requirement."
          skills={job.nice_to_have ?? []}
          tone="muted"
        />
      </motion.section>

      {/* Footer apply CTA — flips to "view application" if one already exists */}
      <motion.section
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="flex flex-col gap-4 rounded-2xl border border-border bg-card p-6 md:flex-row md:items-center md:justify-between"
      >
        <div>
          <div className="font-display text-lg">
            {myApplication ? "You've already applied" : "Ready to apply?"}
          </div>
          <div className="text-sm text-muted-foreground">
            {myApplication
              ? "Track your application status — fit, stage, and next steps live in one place."
              : "Submit your CV once — you'll see your match against this role instantly, and the recruiter is notified the moment you do."}
          </div>
        </div>
        {myApplication ? (
          <AppliedFooterCta application={myApplication} />
        ) : (
        <Link
          to="/c/apply/$id"
          params={{ id: job.id }}
          aria-disabled={job.status !== "open"}
          className={`group inline-flex items-center justify-center gap-1.5 rounded-full px-5 py-2.5 text-sm font-semibold shadow-glow press-tight transition-transform ${
            job.status === "open"
              ? "bg-violet-grad text-accent-foreground hover:scale-[1.03]"
              : "pointer-events-none cursor-not-allowed bg-muted text-muted-foreground"
          }`}
        >
          <Sparkles className="h-4 w-4" />
          {job.status === "open" ? "Submit my application" : "Role closed"}
          {job.status === "open" && <ArrowRight className="h-4 w-4" />}
        </Link>
        )}
      </motion.section>
    </div>
  );
}

function SkillBlock({
  heading,
  subtitle,
  skills,
  tone,
}: {
  heading: string;
  subtitle: string;
  skills: string[];
  tone: "accent" | "muted";
}) {
  return (
    <div className="rounded-2xl border border-border bg-card p-5">
      <div className="text-xs uppercase tracking-widest text-muted-foreground">
        {heading}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">{subtitle}</p>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {skills.length === 0 ? (
          <span className="text-xs italic text-muted-foreground">
            (none listed)
          </span>
        ) : (
          skills.map((s) => (
            <span
              key={s}
              className={
                tone === "accent"
                  ? "rounded-full bg-accent/15 px-2.5 py-1 text-xs text-accent-foreground"
                  : "rounded-full bg-muted px-2.5 py-1 text-xs"
              }
            >
              {s}
            </span>
          ))
        )}
      </div>
    </div>
  );
}

function Chip({
  children,
  tone = "muted",
}: {
  children: React.ReactNode;
  tone?: "muted" | "success";
}) {
  const cls =
    tone === "success"
      ? "bg-success/20 text-success-foreground"
      : "bg-muted text-muted-foreground";
  return (
    <span
      className={`rounded-full px-2.5 py-1 text-[10px] font-medium uppercase tracking-widest ${cls}`}
    >
      {children}
    </span>
  );
}

function prettify(s: string): string {
  if (!s) return s;
  return s
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ─── Already-applied state — hero card + CTAs ──────────────────────────────

const APPLIED_STAGE_LABEL: Record<ApplicationStage, string> = {
  applied: "Submitted",
  approved: "Interview ready",
  interviewed: "Interview captured",
  offer: "Offer extended",
  hired: "Hired",
  rejected: "Closed",
};

function AppliedFitCard({ application }: { application: Application }) {
  const fit = Math.round(application.fit_score * 100);
  return (
    <div className="rounded-2xl border border-accent/40 bg-accent/10 p-4 md:max-w-xs md:text-left">
      <div className="inline-flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-accent">
        <CircleCheck className="h-3 w-3" />
        You've applied · {APPLIED_STAGE_LABEL[application.stage]}
      </div>
      <div className="mt-2 flex items-baseline gap-1.5">
        <div className="font-mono text-4xl text-accent tabular-nums">{fit}</div>
        <div className="text-sm text-muted-foreground">% fit</div>
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-muted/50">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${fit}%` }}
          transition={{ duration: 0.9 }}
          className="h-full bg-violet-grad"
        />
      </div>
    </div>
  );
}

function AppliedHeroCta({ application }: { application: Application }) {
  if (application.stage === "approved") {
    return (
      <Link
        to="/c/interview/$id"
        params={{ id: application.id }}
        className="group press-tight glow-pulse mt-4 inline-flex items-center gap-1.5 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow transition-transform hover:scale-[1.03]"
      >
        <Sparkles className="h-4 w-4" />
        Start your AI interview
        <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
      </Link>
    );
  }
  return (
    <Link
      to="/c/applications"
      className="group press-tight mt-4 inline-flex items-center gap-1.5 rounded-full border border-border bg-card px-4 py-2 text-sm font-semibold transition-transform hover:scale-[1.02] hover:bg-muted"
    >
      View my application
      <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
    </Link>
  );
}

function AppliedFooterCta({ application }: { application: Application }) {
  if (application.stage === "approved") {
    return (
      <Link
        to="/c/interview/$id"
        params={{ id: application.id }}
        className="group press-tight glow-pulse inline-flex items-center justify-center gap-1.5 rounded-full bg-violet-grad px-5 py-2.5 text-sm font-semibold text-accent-foreground shadow-glow transition-transform hover:scale-[1.03]"
      >
        <Sparkles className="h-4 w-4" />
        Start AI interview
        <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
      </Link>
    );
  }
  return (
    <Link
      to="/c/applications"
      className="group press-tight inline-flex items-center justify-center gap-1.5 rounded-full bg-violet-grad px-5 py-2.5 text-sm font-semibold text-accent-foreground shadow-glow transition-transform hover:scale-[1.03]"
    >
      Track my application
      <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
    </Link>
  );
}
