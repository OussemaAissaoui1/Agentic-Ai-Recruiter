import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowLeft, Loader2, Sparkles, Trash2 } from "lucide-react";
import { toast } from "sonner";

import { InterviewReportView } from "@/components/scoring/InterviewReportView";
import { Skeleton } from "@/components/ui/skeleton";
import {
  useApplication,
  useDeleteScoringReport,
  useInterviewsByApplication,
  useRunScoring,
  useScoringReport,
} from "@/lib/queries";

export const Route = createFileRoute("/app/applicants_/$id/report")({
  head: ({ params }) => ({
    meta: [{ title: `Report — ${params.id} — HireFlow` }],
  }),
  component: ReportPage,
});

function ReportPage() {
  const { id: applicationId } = Route.useParams();
  const { data: application } = useApplication(applicationId);
  const { data: interviewList, isLoading: interviewsLoading } =
    useInterviewsByApplication(applicationId);
  const interview = interviewList?.[0];
  const interviewId = interview?.id;

  const { data: report, isLoading: reportLoading } =
    useScoringReport(interviewId);
  const run = useRunScoring();
  const del = useDeleteScoringReport();

  const onRun = async (force: boolean) => {
    if (!interviewId) return;
    try {
      await run.mutateAsync({ interview_id: interviewId, force });
      toast.success(force ? "Report regenerated." : "Report generated.");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  };

  const onDelete = async () => {
    if (!interviewId) return;
    try {
      await del.mutateAsync(interviewId);
      toast.success("Cached report deleted.");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="mx-auto max-w-4xl px-6 py-8">
      <Link
        to="/app/applicants"
        search={{ c: applicationId }}
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="h-4 w-4" />
        Back to applicants
      </Link>

      <div className="mt-4">
        {interviewsLoading ? (
          <Skeleton className="h-64 w-full rounded-2xl" />
        ) : !interview ? (
          <EmptyState
            title="No interview captured"
            description={
              application
                ? `${application.candidate_name || application.candidate_email} hasn't completed the live interview yet. Once they do, you'll be able to score it here.`
                : "This applicant hasn't been interviewed yet."
            }
          />
        ) : reportLoading ? (
          <Skeleton className="h-96 w-full rounded-2xl" />
        ) : !report ? (
          <EmptyState
            title="No report yet"
            description={`Interview captured (${interview.transcript.length} ${interview.transcript.length === 1 ? "turn" : "turns"}). Generate a scoring report to see the per-answer breakdown.`}
            action={
              <button
                onClick={() => onRun(false)}
                disabled={run.isPending || interview.transcript.length === 0}
                className="inline-flex items-center gap-2 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow disabled:opacity-60"
              >
                {run.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Sparkles className="h-4 w-4" />
                )}
                Generate scoring report
              </button>
            }
          />
        ) : (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-end gap-2">
              <button
                onClick={() => onRun(true)}
                disabled={run.isPending}
                className="inline-flex items-center gap-2 rounded-full border border-border px-3 py-1.5 text-xs font-medium hover:bg-muted disabled:opacity-60"
              >
                {run.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Sparkles className="h-3.5 w-3.5" />
                )}
                Regenerate
              </button>
              <button
                onClick={onDelete}
                disabled={del.isPending}
                className="inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium text-muted-foreground hover:text-destructive disabled:opacity-60"
              >
                <Trash2 className="h-3.5 w-3.5" />
                Delete cached
              </button>
            </div>
            <InterviewReportView report={report} />
          </div>
        )}
      </div>
    </div>
  );
}

function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-dashed border-border bg-muted/20 px-6 py-12 text-center">
      <h2 className="font-display text-xl">{title}</h2>
      <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
        {description}
      </p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}
