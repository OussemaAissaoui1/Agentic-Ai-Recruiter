import { motion } from "framer-motion";
import { Database, Inbox, Loader2, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { JDGraphNode } from "@/lib/api";

import { NODE_COLOR } from "./three-cache";

// ---------------------------------------------------------------------------
// CanvasOverlay — full-canvas centered message used for loading / error.
// ---------------------------------------------------------------------------
export function CanvasOverlay({
  text,
  tone = "default",
}: {
  text: string;
  tone?: "default" | "error";
}) {
  return (
    <div
      className={`absolute inset-0 grid place-items-center text-sm ${
        tone === "error" ? "text-rose-500" : "text-muted-foreground"
      }`}
    >
      <div className="max-w-md text-center">{text}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// CanvasEmptyState — shown when there's no graph data yet (instead of a
// blank canvas). Offers the right CTA based on the active data source.
// ---------------------------------------------------------------------------
export function CanvasEmptyState({
  onSeed,
  seeding,
  showSeedCTA,
}: {
  onSeed: () => void;
  seeding: boolean;
  showSeedCTA: boolean;
}) {
  return (
    <div className="absolute inset-0 grid place-items-center p-6">
      <div className="flex max-w-sm flex-col items-center gap-3 text-center">
        <div className="grid h-12 w-12 place-items-center rounded-full bg-muted/40 text-muted-foreground">
          <Inbox className="h-5 w-5" aria-hidden />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-foreground">No graph data yet</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            {showSeedCTA
              ? "Seed the default 100-employee synthetic dataset to explore the graph in 3D, or switch to Custom data to upload your own."
              : "Upload a JSON or zip dataset to populate the graph. Use Augment to merge with existing data, or Replace to start fresh."}
          </p>
        </div>
        {showSeedCTA && (
          <Button onClick={onSeed} disabled={seeding} size="sm" className="min-h-[44px]">
            {seeding ? (
              <>
                <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden />
                Seeding…
              </>
            ) : (
              <>
                <Database className="mr-1.5 h-4 w-4" aria-hidden />
                Seed default data
              </>
            )}
          </Button>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Legend — color key for node types. pointer-events-none so it never steals
// drag/zoom from the canvas.
// ---------------------------------------------------------------------------
export function Legend() {
  return (
    <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap gap-2 rounded-xl border border-border bg-background/80 px-3 py-2 text-[10px] uppercase tracking-wider backdrop-blur-xl">
      {(Object.entries(NODE_COLOR) as Array<[JDGraphNode["type"], string]>).map(
        ([type, color]) => (
          <div key={type} className="flex items-center gap-1.5 text-muted-foreground">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: color }}
              aria-hidden
            />
            {type.replace("_", " ")}
          </div>
        ),
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// NodeDetail — slide-in card showing the selected node's properties.
// ---------------------------------------------------------------------------
export function NodeDetail({
  node,
  onClose,
}: {
  node: JDGraphNode;
  onClose: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 10 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 10 }}
      className="pointer-events-auto absolute right-3 top-3 max-h-[80%] w-72 overflow-y-auto rounded-xl border border-border bg-background/95 p-3 shadow-xl backdrop-blur-xl"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
            {node.type.replace("_", " ")}
          </div>
          <div className="break-words text-sm font-semibold">{node.label}</div>
        </div>
        <button
          onClick={onClose}
          aria-label="Close node detail"
          className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          <X className="h-4 w-4" aria-hidden />
        </button>
      </div>
      {node.props && Object.keys(node.props).length > 0 && (
        <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-xs">
          {Object.entries(node.props).map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-muted-foreground">{k}</dt>
              <dd className="break-words text-foreground">
                {typeof v === "object" ? JSON.stringify(v) : String(v ?? "—")}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </motion.div>
  );
}
