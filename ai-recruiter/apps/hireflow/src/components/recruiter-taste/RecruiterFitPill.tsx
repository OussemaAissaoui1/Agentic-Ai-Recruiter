import { Sparkles } from "lucide-react";

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useRecruiterFitScore } from "@/lib/queries";
import type {
  RecruiterFactorContribution,
  RecruiterTasteConfidence,
} from "@/lib/api";

/**
 * Recruiter-fit signal — a SECOND opinion alongside the existing
 * recommendation chip. Visually distinct on purpose: dashed outline +
 * neutral palette so it never gets confused with the green/yellow/red
 * recommendation pill. Cold-start state is muted with a "learning" hint.
 */
export function RecruiterFitPill({
  applicationId,
  size = "md",
}: {
  applicationId: string;
  size?: "sm" | "md";
}) {
  const { data, isLoading, isError } = useRecruiterFitScore(applicationId);

  const dim =
    size === "sm"
      ? "px-2 py-0.5 text-[10px]"
      : "px-2.5 py-1 text-xs";

  if (isLoading || !data) {
    if (isError) return null;
    return (
      <span
        className={`inline-flex items-center gap-1 rounded-md border border-dashed border-border/60 bg-muted/30 font-medium text-muted-foreground ${dim}`}
      >
        <Sparkles className="h-3 w-3 opacity-50" />
        …
      </span>
    );
  }

  const cold = data.confidence === "cold";
  const palette = paletteFor(data.score, data.confidence);
  const label = labelFor(data.score, data.confidence, data.recruiter_id);

  return (
    <TooltipProvider delayDuration={150}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className={`inline-flex cursor-help items-center gap-1 rounded-md border ${palette.border} ${palette.bg} ${palette.text} font-medium ${dim} ${cold ? "opacity-70" : ""}`}
            data-recruiter-fit-pill=""
          >
            <Sparkles className="h-3 w-3" />
            {label}
          </span>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs space-y-2 p-3">
          <TooltipHeader confidence={data.confidence} score={data.score} />
          <FactorList
            heading="Pulled toward approve"
            factors={data.top_positive}
            sign="+"
          />
          <FactorList
            heading="Pulled toward reject"
            factors={data.top_negative}
            sign="−"
          />
          <ConfidenceFooter confidence={data.confidence} />
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

function paletteFor(
  score: number,
  confidence: RecruiterTasteConfidence,
): { bg: string; text: string; border: string } {
  // Neutral palette so this never collides with the recommendation chip's
  // success/warning/destructive set. Dashed border + slate-tinted bg.
  if (confidence === "cold") {
    return {
      bg: "bg-muted/40",
      text: "text-muted-foreground",
      border: "border-dashed border-border/70",
    };
  }
  if (score >= 0.65) {
    return {
      bg: "bg-accent/15",
      text: "text-accent-foreground",
      border: "border-dashed border-accent/40",
    };
  }
  if (score >= 0.4) {
    return {
      bg: "bg-muted/50",
      text: "text-foreground",
      border: "border-dashed border-border",
    };
  }
  return {
    bg: "bg-muted/40",
    text: "text-muted-foreground",
    border: "border-dashed border-border/60",
  };
}

function labelFor(
  score: number,
  confidence: RecruiterTasteConfidence,
  _recruiterId: string,
): string {
  if (confidence === "cold") return "Learning your taste";
  return `Your fit ${Math.round(score * 100)}%`;
}

function TooltipHeader({
  confidence,
  score,
}: {
  confidence: RecruiterTasteConfidence;
  score: number;
}) {
  return (
    <div className="flex items-baseline justify-between gap-2 border-b border-border/50 pb-1">
      <div className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
        Recruiter fit
      </div>
      <div className="font-mono text-sm font-semibold">
        {confidence === "cold" ? "—" : `${Math.round(score * 100)}%`}
      </div>
    </div>
  );
}

function FactorList({
  heading,
  factors,
  sign,
}: {
  heading: string;
  factors: RecruiterFactorContribution[];
  sign: string;
}) {
  if (factors.length === 0) return null;
  return (
    <div className="space-y-0.5">
      <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
        {heading}
      </div>
      <ul className="space-y-0.5 text-xs">
        {factors.map((f) => (
          <li key={f.label} className="flex items-baseline justify-between gap-2">
            <span className="text-foreground">{f.label}</span>
            <span className="font-mono text-muted-foreground">
              {sign}
              {Math.abs(f.contribution).toFixed(2)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ConfidenceFooter({
  confidence,
}: {
  confidence: RecruiterTasteConfidence;
}) {
  const text = {
    cold: "Cold start — learning your preferences. Score is muted until ~10 decisions.",
    warming: "Warming up — based on a small history; treat as a hint.",
    warm: "Warm — fit reflects your historical approve/reject patterns.",
  }[confidence];
  return (
    <div className="border-t border-border/50 pt-1 text-[10px] italic text-muted-foreground">
      {text}
    </div>
  );
}
