import { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Box,
  CheckCircle2,
  Crosshair,
  FileText,
  Loader2,
  Network as NetworkIcon,
  Square,
  XCircle,
} from "lucide-react";

import type {
  JDGenerateResult,
  JDGraphDump,
  JDGraphNode,
  JDLevel,
} from "@/lib/api";

import { CanvasEmptyState, CanvasOverlay, Legend, NodeDetail } from "./CanvasChrome";
import type { ToolEvent } from "./helpers";
import { JDDraftCard } from "./JDDraftCard";
import { HALO_COLOR, NODE_COLOR, nodeMeshFor } from "./three-cache";

// Lazy-load both engines — Three.js is heavy, no point paying for it on a
// 2D-only session, and vice-versa.
const ForceGraph2D = lazy(() =>
  import("react-force-graph-2d").then((m) => ({ default: m.default })),
);
const ForceGraph3D = lazy(() =>
  import("react-force-graph-3d").then((m) => ({ default: m.default })),
);

type Tab = "graph" | "draft" | "activity";
type Dim = "2d" | "3d";

// ---------------------------------------------------------------------------
// ExplorerStage — tabbed right pane with three modes:
//   • Graph    — the knowledge graph viewer (2D by default, 3D on toggle)
//   • Draft    — the generated JD card, displayed at comfortable reading size
//   • Activity — the tool-call timeline, displayed full-size
//
// Tabs auto-switch on important events: when a JD lands, jump to Draft.
// ---------------------------------------------------------------------------
export function ExplorerStage(props: {
  dump: JDGraphDump | null;
  graphLoading: boolean;
  graphError: string | null;
  isEmpty: boolean;
  source: "default" | "custom";
  uploading: boolean;
  onSeedDefault: () => void;
  // graph interaction
  highlighted: Set<string>;
  selected: JDGraphNode | null;
  setSelected: (n: JDGraphNode | null) => void;
  reducedMotion: boolean | null;
  // JD-side
  generated: JDGenerateResult | null;
  generationError: string | null;
  toolEvents: ToolEvent[];
  running: boolean;
  roleTitle: string;
  level: JDLevel;
  teamId: string;
}) {
  const [tab, setTab] = useState<Tab>("graph");
  const [dim, setDim] = useState<Dim>("2d");
  const fgRef = useRef<any>(null);

  // Auto-switch to Draft once a JD lands, and back to Graph if user clears.
  useEffect(() => {
    if (props.generated) setTab("draft");
  }, [props.generated]);
  useEffect(() => {
    if (props.running) setTab("activity");
  }, [props.running]);

  // Track container size so the canvas matches its parent (avoids the
  // "graph rendered off-center" bug from defaulting to window size).
  const boxRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      setSize((p) => {
        const w = Math.round(r.width);
        const h = Math.round(r.height);
        return p.width === w && p.height === h ? p : { width: w, height: h };
      });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Re-frame the camera/view when the container resizes or the active tab
  // changes back to graph (the canvas is hidden when other tabs are active).
  useEffect(() => {
    if (tab !== "graph" || size.width <= 0) return;
    const fg = fgRef.current;
    if (!fg || typeof fg.zoomToFit !== "function") return;
    const t = window.setTimeout(
      () => fg.zoomToFit(props.reducedMotion ? 0 : 400, 60),
      120,
    );
    return () => window.clearTimeout(t);
  }, [tab, size.width, size.height, dim, props.reducedMotion]);

  const graphData = useMemo(
    () =>
      props.dump
        ? {
            nodes: props.dump.nodes.map((n) => ({
              ...n,
              _halo: props.highlighted.has(n.id),
            })),
            links: props.dump.links,
          }
        : { nodes: [], links: [] },
    [props.dump, props.highlighted],
  );

  const hasJD = !!props.generated;
  const hasActivity = props.toolEvents.length > 0 || props.running;

  return (
    <section className="flex h-full min-h-0 flex-col overflow-hidden rounded-3xl border border-border bg-card/40 shadow-sm">
      {/* Tab strip + per-tab right-side controls */}
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-border bg-background/40 px-3 py-2 backdrop-blur">
        <div role="tablist" aria-label="Stage view" className="inline-flex gap-1">
          <TabButton active={tab === "graph"} onClick={() => setTab("graph")}>
            <NetworkIcon className="h-3.5 w-3.5" aria-hidden />
            Graph
          </TabButton>
          <TabButton
            active={tab === "draft"}
            onClick={() => setTab("draft")}
            disabled={!hasJD}
            badge={hasJD ? "●" : undefined}
          >
            <FileText className="h-3.5 w-3.5" aria-hidden />
            Draft JD
          </TabButton>
          <TabButton
            active={tab === "activity"}
            onClick={() => setTab("activity")}
            disabled={!hasActivity}
            badge={
              props.toolEvents.length > 0
                ? String(props.toolEvents.length)
                : undefined
            }
          >
            <Activity className="h-3.5 w-3.5" aria-hidden />
            Activity
          </TabButton>
        </div>

        <div className="ml-auto flex items-center gap-2">
          {tab === "graph" && !props.isEmpty && !props.graphLoading && !props.graphError && (
            <>
              <DimToggle dim={dim} setDim={setDim} />
              <button
                type="button"
                onClick={() => {
                  const fg = fgRef.current;
                  if (fg && typeof fg.zoomToFit === "function") {
                    fg.zoomToFit(props.reducedMotion ? 0 : 500, 60);
                  }
                }}
                aria-label="Reset camera"
                title="Reset camera"
                className="inline-flex h-9 items-center gap-1.5 rounded-lg border border-border bg-background px-3 text-xs font-medium text-foreground transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                <Crosshair className="h-3.5 w-3.5" aria-hidden />
                Reset view
              </button>
            </>
          )}
        </div>
      </div>

      {/* Active panel */}
      <div className="relative min-h-0 flex-1 overflow-hidden">
        {/* Graph */}
        <div
          ref={boxRef}
          className={`absolute inset-0 ${tab === "graph" ? "" : "hidden"}`}
          role="img"
          aria-label={
            props.dump
              ? `Knowledge graph with ${props.dump.node_count.toLocaleString()} nodes and ${props.dump.link_count.toLocaleString()} links`
              : "Knowledge graph (empty)"
          }
          style={{
            backgroundImage:
              "radial-gradient(ellipse at center, var(--card) 0%, var(--background) 75%)",
          }}
        >
          {props.graphLoading && <CanvasOverlay text="Loading graph…" />}
          {props.graphError && !props.graphLoading && (
            <CanvasOverlay text={props.graphError} tone="error" />
          )}
          {props.isEmpty && (
            <CanvasEmptyState
              onSeed={props.onSeedDefault}
              seeding={props.uploading}
              showSeedCTA={props.source === "default"}
            />
          )}
          {!props.graphError && !props.isEmpty && props.dump && size.width > 0 && (
            <Suspense fallback={<CanvasOverlay text="Loading graph engine…" />}>
              {dim === "2d" ? (
                <ForceGraph2D
                  ref={fgRef}
                  width={size.width}
                  height={size.height}
                  graphData={graphData}
                  backgroundColor="rgba(0,0,0,0)"
                  nodeLabel={(n: any) => `${n.label} (${n.type})`}
                  nodeColor={(n: any) =>
                    n._halo ? HALO_COLOR : NODE_COLOR[n.type as keyof typeof NODE_COLOR] ?? "#888"
                  }
                  nodeRelSize={5}
                  linkColor={() => "rgba(148, 163, 184, 0.4)"}
                  linkWidth={(l: any) =>
                    typeof l.proficiency === "number"
                      ? Math.max(0.5, l.proficiency / 1.5)
                      : typeof l.frequency === "number"
                      ? Math.max(0.5, Math.log10(l.frequency + 1) + 0.5)
                      : 0.6
                  }
                  cooldownTicks={props.reducedMotion ? 0 : 200}
                  onNodeClick={(n: any) => props.setSelected(n as JDGraphNode)}
                  onEngineStop={() => {
                    const fg = fgRef.current;
                    if (fg && typeof fg.zoomToFit === "function") {
                      fg.zoomToFit(props.reducedMotion ? 0 : 600, 60);
                    }
                  }}
                  nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, scale: number) => {
                    // Custom 2D node renderer: filled circle + label below at
                    // higher zoom levels. Halos render a ring around the dot.
                    const fill = node._halo
                      ? HALO_COLOR
                      : NODE_COLOR[node.type as keyof typeof NODE_COLOR] ?? "#888";
                    const r = node._halo ? 6 : 4;
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
                    ctx.fillStyle = fill;
                    ctx.fill();
                    if (node._halo) {
                      ctx.beginPath();
                      ctx.arc(node.x, node.y, r + 3, 0, Math.PI * 2);
                      ctx.strokeStyle = HALO_COLOR;
                      ctx.lineWidth = 1.5;
                      ctx.stroke();
                    }
                    if (scale > 2.5) {
                      const label = String(node.label || "").slice(0, 24);
                      ctx.font = "10px system-ui, sans-serif";
                      ctx.textAlign = "center";
                      ctx.textBaseline = "top";
                      ctx.fillStyle = "rgba(148,163,184,0.95)";
                      ctx.fillText(label, node.x, node.y + r + 2);
                    }
                  }}
                />
              ) : (
                <ForceGraph3D
                  ref={fgRef}
                  width={size.width}
                  height={size.height}
                  graphData={graphData}
                  backgroundColor="rgba(0,0,0,0)"
                  nodeLabel={(n: any) => `${n.label} (${n.type})`}
                  nodeColor={(n: any) =>
                    n._halo ? HALO_COLOR : NODE_COLOR[n.type as keyof typeof NODE_COLOR] ?? "#888"
                  }
                  nodeRelSize={5}
                  linkColor={() => "rgba(148, 163, 184, 0.35)"}
                  linkWidth={(l: any) =>
                    typeof l.proficiency === "number"
                      ? Math.max(0.3, l.proficiency / 2)
                      : typeof l.frequency === "number"
                      ? Math.max(0.3, Math.log10(l.frequency + 1) + 0.5)
                      : 0.4
                  }
                  linkOpacity={0.6}
                  nodeOpacity={0.95}
                  onNodeClick={(n: any) => props.setSelected(n as JDGraphNode)}
                  cooldownTicks={props.reducedMotion ? 0 : 200}
                  enableNodeDrag={!props.reducedMotion}
                  onEngineStop={() => {
                    const fg = fgRef.current;
                    if (fg && typeof fg.zoomToFit === "function") {
                      fg.zoomToFit(props.reducedMotion ? 0 : 600, 60);
                    }
                  }}
                  nodeThreeObject={(node: any) => nodeMeshFor(node.type, !!node._halo)}
                />
              )}
            </Suspense>
          )}
          {props.selected && (
            <NodeDetail node={props.selected} onClose={() => props.setSelected(null)} />
          )}
          {!props.isEmpty && <Legend />}
        </div>

        {/* Draft */}
        {tab === "draft" && (
          <div className="absolute inset-0 overflow-y-auto p-5 sm:p-8">
            <div className="mx-auto max-w-3xl">
              {props.generated ? (
                <JDDraftCard
                  result={props.generated}
                  roleTitle={props.roleTitle}
                  level={props.level}
                  teamId={props.teamId}
                />
              ) : (
                <EmptyTab
                  icon={<FileText className="h-5 w-5" aria-hidden />}
                  title="No draft yet"
                  message="Fill in the role on the left and hit Generate."
                />
              )}
              {props.generationError && (
                <div
                  role="alert"
                  className="mt-4 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-500"
                >
                  {props.generationError}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Activity */}
        {tab === "activity" && (
          <div className="absolute inset-0 overflow-y-auto p-5 sm:p-8">
            <div className="mx-auto max-w-2xl">
              <ActivityTimeline events={props.toolEvents} running={props.running} />
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Tab pill
// ---------------------------------------------------------------------------
function TabButton({
  active,
  disabled,
  onClick,
  children,
  badge,
}: {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: React.ReactNode;
  badge?: string;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex min-h-[36px] items-center gap-1.5 rounded-lg px-3 text-xs font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
        active
          ? "bg-accent/15 text-accent shadow-sm"
          : disabled
          ? "cursor-not-allowed text-muted-foreground/50"
          : "text-muted-foreground hover:bg-muted hover:text-foreground"
      }`}
    >
      {children}
      {badge && (
        <span
          className={`rounded-full px-1.5 py-px text-[10px] tabular-nums ${
            active ? "bg-accent/20 text-accent" : "bg-muted text-muted-foreground"
          }`}
        >
          {badge}
        </span>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// 2D / 3D dimension toggle
// ---------------------------------------------------------------------------
function DimToggle({ dim, setDim }: { dim: Dim; setDim: (d: Dim) => void }) {
  return (
    <div
      role="group"
      aria-label="Graph dimension"
      className="inline-flex rounded-lg border border-border bg-background p-0.5"
    >
      {(["2d", "3d"] as const).map((d) => (
        <button
          key={d}
          type="button"
          onClick={() => setDim(d)}
          aria-pressed={dim === d}
          title={d === "2d" ? "Flat layout (faster, easier to read)" : "3D layout (depth)"}
          className={`inline-flex h-8 items-center gap-1 rounded-md px-2.5 text-[11px] font-semibold uppercase tracking-wider transition ${
            dim === d
              ? "bg-accent/15 text-accent"
              : "text-muted-foreground hover:text-foreground"
          }`}
        >
          {d === "2d" ? (
            <Square className="h-3 w-3" aria-hidden />
          ) : (
            <Box className="h-3 w-3" aria-hidden />
          )}
          {d}
        </button>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Activity tab — bigger, more legible version of the tool-call log
// ---------------------------------------------------------------------------
function ActivityTimeline({
  events,
  running,
}: {
  events: ToolEvent[];
  running: boolean;
}) {
  if (events.length === 0 && !running) {
    return (
      <EmptyTab
        icon={<Activity className="h-5 w-5" aria-hidden />}
        title="No activity yet"
        message="The agent's tool calls will stream here as it works."
      />
    );
  }
  return (
    <div>
      <h2 className="text-lg font-semibold tracking-tight text-foreground">
        Agent activity
      </h2>
      <p className="mt-1 text-xs text-muted-foreground">
        Every tool call the agent makes, in order. Yellow halos on graph nodes
        light up as data is fetched.
      </p>
      <ol className="mt-5 space-y-3">
        {events.map((ev, i) => (
          <li
            key={`${ev.tool}-${i}`}
            className="flex items-start gap-3 rounded-xl border border-border bg-background p-3"
          >
            <div className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg">
              {ev.ok === null ? (
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" aria-hidden />
              ) : ev.ok ? (
                <span className="grid h-8 w-8 place-items-center rounded-lg bg-emerald-500/15 text-emerald-500">
                  <CheckCircle2 className="h-4 w-4" aria-hidden />
                </span>
              ) : (
                <span className="grid h-8 w-8 place-items-center rounded-lg bg-rose-500/15 text-rose-500">
                  <XCircle className="h-4 w-4" aria-hidden />
                </span>
              )}
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-3">
                <span className="font-mono text-sm font-semibold text-foreground">
                  {ev.tool}
                </span>
                <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
                  step {String(i + 1).padStart(2, "0")}
                </span>
              </div>
              {ev.summary && (
                <div className="mt-1 text-xs text-muted-foreground">{ev.summary}</div>
              )}
            </div>
          </li>
        ))}
        {running && events.length > 0 && events[events.length - 1].ok !== null && (
          <li className="flex items-center gap-3 rounded-xl border border-dashed border-border bg-muted/20 p-3 text-xs text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            Thinking…
          </li>
        )}
      </ol>
    </div>
  );
}

function EmptyTab({
  icon,
  title,
  message,
}: {
  icon: React.ReactNode;
  title: string;
  message: string;
}) {
  return (
    <div className="grid place-items-center pt-12 text-center">
      <div className="grid h-12 w-12 place-items-center rounded-full bg-muted/40 text-muted-foreground">
        {icon}
      </div>
      <h3 className="mt-3 text-sm font-semibold text-foreground">{title}</h3>
      <p className="mt-1 max-w-xs text-xs text-muted-foreground">{message}</p>
    </div>
  );
}
