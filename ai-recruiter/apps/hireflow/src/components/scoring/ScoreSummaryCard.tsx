import { Link } from "@tanstack/react-router";
import { Loader2, Sparkles, RefreshCw, ExternalLink } from "lucide-react";
import { toast } from "sonner";

import { RecommendationChip } from "./RecommendationChip";
import {
  useInterviewsByApplication,
  useRunScoring,
  useScoringReport,
} from "@/lib/queries";

export function ScoreSummaryCard({ applicationId }: { applicationId: string }) {
  const { data: interviewList, isLoading: interviewsLoading } =
    useInterviewsByApplication(applicationId);
  const interview = interviewList?.[0]; // most recent
  const interviewId = interview?.id;

  const { data: report, isLoading: reportLoading } =
    useScoringReport(interviewId);
  const run = useRunScoring();

  const onRun = async (force: boolean) => {
    if (!interviewId) return;
    try {
      await run.mutateAsync({ interview_id: interviewId, force });
      toast.success(force ? "Report regenerated." : "Report generated.");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="rounded-2xl border border-border bg-background p-4">
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-widest text-muted-foreground">
          Interview report
        </div>
        {report && (
          <button
            onClick={() => onRun(true)}
            disabled={run.isPending}
            className="inline-flex items-center gap-1 rounded-full px-2 py-1 text-[11px] text-muted-foreground hover:text-foreground disabled:opacity-50"
            title="Regenerate report"
          >
            {run.isPending ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <RefreshCw className="h-3 w-3" />
            )}
            Regenerate
          </button>
        )}
      </div>

      {interviewsLoading ? (
        <div className="mt-3 text-xs text-muted-foreground">
          Checking for interview…
        </div>
      ) : !interview ? (
        <p className="mt-2 text-xs italic text-muted-foreground">
          No completed interview yet. Once the candidate finishes the live
          interview, you'll be able to score it here.
        </p>
      ) : reportLoading ? (
        <div className="mt-3 text-xs text-muted-foreground">
          Loading report…
        </div>
      ) : !report ? (
        <div className="mt-3 space-y-3">
          <p className="text-sm text-muted-foreground">
            Interview captured ({interview.transcript.length}{" "}
            {interview.transcript.length === 1 ? "turn" : "turns"}). No report
            yet.
          </p>
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
        </div>
      ) : (
        <div className="mt-3 space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <RecommendationChip value={report.overall.recommendation} />
            <div className="text-xs text-muted-foreground">
              Technical{" "}
              <span className="font-mono font-semibold text-foreground">
                {report.overall.technical_avg.toFixed(1)}/5
              </span>
            </div>
            <div className="text-xs text-muted-foreground">
              Coherence{" "}
              <span className="font-mono font-semibold text-foreground">
                {report.overall.coherence_avg.toFixed(1)}/5
              </span>
            </div>
          </div>

          {report.overall.summary && (
            <p className="text-sm leading-relaxed text-foreground line-clamp-3">
              {report.overall.summary}
            </p>
          )}

          <Link
            to="/app/applicants/$id/report"
            params={{ id: applicationId }}
            search={{ c: undefined }}
            className="inline-flex items-center gap-1 text-xs font-medium text-accent hover:underline"
          >
            Open full report
            <ExternalLink className="h-3 w-3" />
          </Link>
        </div>
      )}
    </div>
  );
}
