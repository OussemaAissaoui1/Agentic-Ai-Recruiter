import { Network, Sparkles } from "lucide-react";

// ---------------------------------------------------------------------------
// ExplorerHeader — hero strip for the studio layout.
//
// Accent-forward gradient, big title, inline stats, a small "Graph-RAG" pill
// to anchor the brand. Deliberately taller than a typical app header so the
// page feels like its own destination instead of a sub-tool.
// ---------------------------------------------------------------------------
export function ExplorerHeader({
  nodeCount,
  linkCount,
  familyCount,
  source,
  status,
}: {
  nodeCount: number;
  linkCount: number;
  familyCount: number;
  source: "default" | "custom";
  status: "loading" | "ready" | "error" | "empty";
}) {
  return (
    <header className="relative overflow-hidden rounded-3xl border border-border bg-gradient-to-br from-accent/15 via-accent/5 to-background px-6 py-5 sm:px-8 sm:py-6">
      {/* Decorative dot grid in the background — adds texture without
          competing with content. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-[0.05]"
        style={{
          backgroundImage:
            "radial-gradient(currentColor 1px, transparent 1px)",
          backgroundSize: "20px 20px",
        }}
      />

      <div className="relative flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between sm:gap-8">
        <div className="flex items-start gap-4">
          <div className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-gradient-to-br from-accent to-accent/60 text-background shadow-lg shadow-accent/25">
            <Network className="h-6 w-6" aria-hidden />
          </div>
          <div className="min-w-0">
            <div className="inline-flex items-center gap-1.5 rounded-full border border-accent/30 bg-accent/10 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-accent">
              <Sparkles className="h-3 w-3" aria-hidden />
              Graph-RAG studio
            </div>
            <h1 className="mt-2 text-3xl font-semibold leading-tight tracking-tight text-foreground sm:text-4xl">
              Compose grounded JDs
            </h1>
            <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-muted-foreground sm:text-[15px]">
              Every claim is traceable back to a node in your employee skill
              graph. No hallucinations, no rockstars — just evidence.
            </p>
          </div>
        </div>

        {/* Right cluster: stats + status. Wraps below on small screens. */}
        <div className="flex flex-wrap items-end gap-x-6 gap-y-3 sm:flex-col sm:items-end sm:gap-y-2">
          <StatusBadge kind={status} />
          <div className="flex items-baseline gap-5 text-right">
            <Stat label="Nodes" value={nodeCount} />
            <Stat label="Links" value={linkCount} />
            <Stat label="Families" value={familyCount} />
          </div>
          <div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
            {source === "default" ? "Default synthetic dataset" : "Custom dataset"}
          </div>
        </div>
      </div>
    </header>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex flex-col items-end">
      <span className="font-mono text-xl font-semibold tabular-nums leading-none text-foreground sm:text-2xl">
        {value.toLocaleString()}
      </span>
      <span className="mt-1 text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </span>
    </div>
  );
}

function StatusBadge({ kind }: { kind: "loading" | "ready" | "error" | "empty" }) {
  const config = {
    loading: { color: "bg-amber-500",   ring: "ring-amber-500/30",   label: "Loading graph",  pulse: true },
    ready:   { color: "bg-emerald-500", ring: "ring-emerald-500/30", label: "Graph ready",    pulse: false },
    error:   { color: "bg-rose-500",    ring: "ring-rose-500/30",    label: "Graph error",    pulse: false },
    empty:   { color: "bg-slate-400",   ring: "ring-slate-400/30",   label: "No graph yet",   pulse: false },
  }[kind];
  return (
    <span className={`inline-flex items-center gap-2 rounded-full bg-background px-3 py-1 text-[11px] font-medium text-foreground ring-1 ${config.ring}`}>
      <span className="relative inline-flex h-2 w-2">
        {config.pulse && (
          <span aria-hidden className={`absolute inset-0 animate-ping rounded-full ${config.color} opacity-75`} />
        )}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${config.color}`} />
      </span>
      {config.label}
    </span>
  );
}
