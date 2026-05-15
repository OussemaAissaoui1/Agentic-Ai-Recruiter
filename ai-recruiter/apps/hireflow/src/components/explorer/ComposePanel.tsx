import {
  ArrowRight,
  Briefcase,
  Building2,
  CheckCircle2,
  CircleSlash,
  Loader2,
  Play,
  UserCheck,
} from "lucide-react";
import { Link } from "@tanstack/react-router";

import { Button } from "@/components/ui/button";
import type { JDLevel } from "@/lib/api";

import type { ApprovalState } from "./helpers";

// ---------------------------------------------------------------------------
// ComposePanel — left workspace in the studio layout.
//
// Houses:
//   1. The role-input form (the "address bar" of the studio)
//   2. The primary CTA (Generate)
//   3. Once a JD is generated: the Approve/Reject row + post-approval state
//
// The form is always visible (not collapsed behind a toggle) — the user's
// most frequent action is "tweak the role and regenerate".
// ---------------------------------------------------------------------------
export function ComposePanel(props: {
  // form
  roleTitle: string;
  setRoleTitle: (v: string) => void;
  level: JDLevel;
  setLevel: (v: JDLevel) => void;
  teamId: string;
  setTeamId: (v: string) => void;
  // state
  running: boolean;
  hasGenerated: boolean;
  approval: ApprovalState;
  // actions
  onRun: () => void;
  onApprove: () => void;
  onOpenReject: () => void;
}) {
  const canGenerate = !!props.roleTitle.trim() && !props.running;

  return (
    <aside className="flex h-full flex-col gap-4 overflow-y-auto">
      {/* Compose card */}
      <div className="rounded-2xl border border-border bg-card/40 p-5 shadow-sm backdrop-blur-xl">
        <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
          <span className="grid h-5 w-5 place-items-center rounded-md bg-accent/15 text-accent">
            01
          </span>
          Compose
        </div>
        <h2 className="mt-3 text-lg font-semibold leading-tight text-foreground">
          Tell us about the role
        </h2>
        <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
          Pick a title and level. The agent will look up real employees with
          this profile and ground the JD in their skills.
        </p>

        <div className="mt-5 space-y-4">
          <Field
            label="Role title"
            icon={<Briefcase className="h-3.5 w-3.5" aria-hidden />}
          >
            <input
              value={props.roleTitle}
              onChange={(e) => props.setRoleTitle(e.target.value)}
              placeholder="Senior Backend Engineer"
              className="min-h-[44px] w-full rounded-lg border border-border bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            />
          </Field>

          <Field
            label="Level"
            icon={<UserCheck className="h-3.5 w-3.5" aria-hidden />}
          >
            <div className="flex gap-1.5">
              {(["junior", "mid", "senior"] as const).map((lv) => (
                <button
                  key={lv}
                  type="button"
                  onClick={() => props.setLevel(lv)}
                  aria-pressed={props.level === lv}
                  className={`min-h-[44px] flex-1 rounded-lg border text-xs font-medium capitalize transition ${
                    props.level === lv
                      ? "border-accent bg-accent/10 text-accent shadow-sm"
                      : "border-border bg-background text-muted-foreground hover:border-accent/40 hover:text-foreground"
                  }`}
                >
                  {lv}
                </button>
              ))}
            </div>
          </Field>

          <Field
            label="Team"
            optional
            icon={<Building2 className="h-3.5 w-3.5" aria-hidden />}
          >
            <input
              value={props.teamId}
              onChange={(e) => props.setTeamId(e.target.value)}
              placeholder="team_platform"
              className="min-h-[44px] w-full rounded-lg border border-border bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            />
          </Field>
        </div>

        <Button
          onClick={props.onRun}
          disabled={!canGenerate}
          size="lg"
          className="mt-5 min-h-[52px] w-full bg-gradient-to-br from-accent to-accent/80 text-base font-semibold text-background shadow-md shadow-accent/20 transition hover:shadow-accent/30 disabled:from-muted disabled:to-muted disabled:text-muted-foreground disabled:shadow-none"
        >
          {props.running ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" aria-hidden />
              Generating…
            </>
          ) : (
            <>
              <Play className="mr-2 h-5 w-5" aria-hidden />
              {props.hasGenerated ? "Regenerate" : "Generate JD"}
            </>
          )}
        </Button>
      </div>

      {/* Approve / Reject card (only when there's something to approve) */}
      {props.hasGenerated && (
        <div className="rounded-2xl border border-border bg-card/40 p-5 shadow-sm backdrop-blur-xl">
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
            <span className="grid h-5 w-5 place-items-center rounded-md bg-accent/15 text-accent">
              02
            </span>
            Decide
          </div>
          <h2 className="mt-3 text-lg font-semibold leading-tight text-foreground">
            Ship it, or send feedback
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            Approving creates a real Job under <code className="rounded bg-muted px-1 py-px text-[10px]">/app/jobs</code>.
            Rejecting feeds your reason back into the next generation.
          </p>

          <ApprovalActions
            approval={props.approval}
            onApprove={props.onApprove}
            onOpenReject={props.onOpenReject}
          />
        </div>
      )}
    </aside>
  );
}

function Field({
  label,
  optional,
  icon,
  children,
}: {
  label: string;
  optional?: boolean;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        {icon}
        {label}
        {optional && (
          <span className="text-[9px] font-normal normal-case tracking-normal text-muted-foreground/70">
            (optional)
          </span>
        )}
      </span>
      {children}
    </label>
  );
}

function ApprovalActions({
  approval,
  onApprove,
  onOpenReject,
}: {
  approval: ApprovalState;
  onApprove: () => void;
  onOpenReject: () => void;
}) {
  if (approval.kind === "posted") {
    return (
      <div
        role="status"
        aria-live="polite"
        className="mt-4 flex flex-col gap-2 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-xs"
      >
        <div className="flex items-center gap-2 text-emerald-500">
          <CheckCircle2 className="h-4 w-4" aria-hidden />
          <span className="font-medium">Posted to /jobs</span>
        </div>
        <Link
          to="/app/jobs/$id"
          params={{ id: approval.jobId }}
          className="inline-flex min-h-[44px] items-center justify-center gap-1.5 rounded-md border border-emerald-500/30 bg-background px-3 text-emerald-500 transition hover:bg-emerald-500/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500"
        >
          <Briefcase className="h-3.5 w-3.5" aria-hidden />
          View job
          <ArrowRight className="h-3.5 w-3.5" aria-hidden />
        </Link>
      </div>
    );
  }
  return (
    <>
      {approval.kind === "error" && (
        <div
          role="alert"
          className="mt-3 rounded-lg border border-rose-500/30 bg-rose-500/10 p-2 text-xs text-rose-500"
        >
          Couldn't post: {approval.message}
        </div>
      )}
      <div className="mt-4 flex gap-2">
        <Button
          onClick={onApprove}
          disabled={approval.kind === "posting"}
          variant="default"
          className="min-h-[44px] flex-1"
        >
          {approval.kind === "posting" ? (
            <>
              <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden />
              Posting…
            </>
          ) : (
            <>
              <CheckCircle2 className="mr-1.5 h-4 w-4" aria-hidden />
              {approval.kind === "error" ? "Retry post" : "Approve & post"}
            </>
          )}
        </Button>
        <Button
          onClick={onOpenReject}
          disabled={approval.kind === "posting"}
          variant="outline"
          className="min-h-[44px] flex-1"
        >
          <CircleSlash className="mr-1.5 h-4 w-4" aria-hidden />
          Reject
        </Button>
      </div>
    </>
  );
}
