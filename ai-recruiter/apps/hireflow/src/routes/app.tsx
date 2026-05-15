import { createFileRoute } from "@tanstack/react-router";
import { HRShell } from "@/components/Shell";

export const Route = createFileRoute("/app")({
  component: HRShell,
});
