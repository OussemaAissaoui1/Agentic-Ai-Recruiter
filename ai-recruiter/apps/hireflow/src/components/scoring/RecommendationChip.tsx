import type { Recommendation } from "@/lib/api";

const STYLES: Record<Recommendation, string> = {
  strong_hire: "bg-success/25 text-success-foreground",
  hire: "bg-success/15 text-success-foreground",
  lean_hire: "bg-warning/15 text-warning-foreground",
  no_hire: "bg-destructive/15 text-destructive",
};

const LABELS: Record<Recommendation, string> = {
  strong_hire: "Strong hire",
  hire: "Hire",
  lean_hire: "Lean hire",
  no_hire: "No hire",
};

export function RecommendationChip({
  value,
  size = "md",
}: {
  value: Recommendation;
  size?: "sm" | "md";
}) {
  const cls = STYLES[value];
  const dim =
    size === "sm"
      ? "px-2 py-0.5 text-[10px]"
      : "px-2.5 py-1 text-xs";
  return (
    <span
      className={`inline-flex items-center rounded-full font-medium ${dim} ${cls}`}
    >
      {LABELS[value]}
    </span>
  );
}
