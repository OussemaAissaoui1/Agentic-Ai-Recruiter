import {
  useInterviewsByApplication,
  useScoringReport,
} from "@/lib/queries";
import { RecommendationChip } from "./RecommendationChip";

/**
 * Row-level chip that lazily fetches the latest interview's report for a
 * candidate. Only enabled for interviewed candidates so the list view
 * doesn't fan out a request per row across the whole table.
 */
export function RowRecommendation({
  applicationId,
  enabled,
}: {
  applicationId: string;
  enabled: boolean;
}) {
  const { data: interviewList } = useInterviewsByApplication(
    enabled ? applicationId : undefined,
  );
  const interviewId = interviewList?.[0]?.id;
  const { data: report } = useScoringReport(
    enabled ? interviewId : undefined,
  );

  if (!enabled || !report) return null;
  return <RecommendationChip value={report.overall.recommendation} size="sm" />;
}
