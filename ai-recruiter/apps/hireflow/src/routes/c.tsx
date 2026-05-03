import { createFileRoute } from "@tanstack/react-router";
import { CandidateShell } from "@/components/Shell";

export const Route = createFileRoute("/c")({
  component: CandidateShell,
});
