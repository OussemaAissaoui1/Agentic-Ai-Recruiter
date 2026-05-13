import { createFileRoute } from "@tanstack/react-router";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleSlash,
  Database,
  FileUp,
  Loader2,
  Play,
  RotateCcw,
  Sparkles,
  Upload,
  X,
  XCircle,
} from "lucide-react";
import {
  forwardRef,
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import * as THREE from "three";

import { Button } from "@/components/ui/button";
import {
  api,
  type JDGenerateResult,
  type JDGraphDump,
  type JDGraphNode,
  type JDLevel,
  type JDRejectionCheck,
  type JDStreamEvent,
} from "@/lib/api";

// react-force-graph-3d pulls in Three.js and the d3-force engine; lazy-load
// so the bundle isn't paid by users who never visit /app/explorer.
const ForceGraph3D = lazy(() =>
  import("react-force-graph-3d").then((mod) => ({ default: mod.default })),
);

export const Route = createFileRoute("/app/explorer")({
  head: () => ({ meta: [{ title: "Graph Explorer — HireFlow" }] }),
  component: Explorer,
});

// ---------------------------------------------------------------------------
// Node-type colors (kept in sync with the dump.py NODE_TYPE_BY_LABEL keys)
// ---------------------------------------------------------------------------
const NODE_COLOR: Record<JDGraphNode["type"], string> = {
  employee:        "#a78bfa", // violet-400
  role:            "#5eead4", // teal-300 (mint-ish)
  skill:           "#fbbf24", // amber-400
  team:            "#94a3b8", // slate-400
  education:       "#fdba74", // orange-300
  prior_company:   "#fb7185", // rose-400
  job_description: "#c084fc", // violet-400 / brighter
  rejection:       "#ef4444", // red-500
};

const HALO_COLOR = "#facc15"; // yellow-400
const HALO_DURATION_MS = 3500;

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
function Explorer() {
  const reducedMotion = useReducedMotion();
  const fgRef = useRef<any>(null);

  // ── graph data ─────────────────────────────────────────────────────────
  const [dump, setDump] = useState<JDGraphDump | null>(null);
  const [graphLoading, setGraphLoading] = useState(true);
  const [graphError, setGraphError] = useState<string | null>(null);

  // ── filter state ───────────────────────────────────────────────────────
  const [familyFilter, setFamilyFilter] = useState<string | "all">("all");
  const [includeRejections, setIncludeRejections] = useState(true);

  // ── selection ─────────────────────────────────────────────────────────
  const [selected, setSelected] = useState<JDGraphNode | null>(null);

  // ── data source ────────────────────────────────────────────────────────
  type Source = "default" | "custom";
  const [source, setSource] = useState<Source>("default");
  const [uploadStatus, setUploadStatus] = useState<
    | null
    | { kind: "ok"; counts: Record<string, number>; mode: string }
    | { kind: "err"; message: string }
  >(null);
  const [uploading, setUploading] = useState(false);

  // ── agent drawer ──────────────────────────────────────────────────────
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [roleTitle, setRoleTitle] = useState("Senior Backend Engineer");
  const [level, setLevel] = useState<JDLevel>("senior");
  const [teamId, setTeamId] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [toolEvents, setToolEvents] = useState<
    Array<{ tool: string; ok: boolean | null; summary?: string }>
  >([]);
  const [generated, setGenerated] = useState<JDGenerateResult | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);

  // ── reject modal ──────────────────────────────────────────────────────
  const [rejectOpen, setRejectOpen] = useState(false);
  const [rejectText, setRejectText] = useState("");
  const [rejectCats, setRejectCats] = useState<string[]>([]);
  const [rejectBusy, setRejectBusy] = useState(false);

  // ── highlighted nodes (yellow halo) ───────────────────────────────────
  const [highlighted, setHighlighted] = useState<Set<string>>(new Set());
  const haloTimers = useRef<Map<string, number>>(new Map());

  // ── load graph dump ───────────────────────────────────────────────────
  const fetchDump = useCallback(
    async (signal?: AbortSignal) => {
      setGraphLoading(true);
      setGraphError(null);
      try {
        const data = await api.jdGen.dump(
          {
            filter: familyFilter === "all" ? undefined : `role_family:${familyFilter}`,
            include_rejections: includeRejections,
          },
          { signal },
        );
        setDump(data);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        // 501 means Phase 6 hasn't shipped on the backend — show a nicer message.
        if (/501/.test(msg) || /not implemented/i.test(msg)) {
          setGraphError(
            "Graph dump endpoint not enabled yet. Set up Neo4j and seed default data.",
          );
        } else {
          setGraphError(msg);
        }
      } finally {
        setGraphLoading(false);
      }
    },
    [familyFilter, includeRejections],
  );

  useEffect(() => {
    const ctrl = new AbortController();
    fetchDump(ctrl.signal);
    return () => ctrl.abort();
  }, [fetchDump]);

  // ── upload handler ────────────────────────────────────────────────────
  const onFileChosen = useCallback(
    async (file: File, mode: "augment" | "replace") => {
      setUploading(true);
      setUploadStatus(null);
      try {
        const res = await api.jdGen.upload(file, mode);
        setUploadStatus({ kind: "ok", counts: res.counts, mode });
        await fetchDump();
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setUploadStatus({ kind: "err", message: msg });
      } finally {
        setUploading(false);
      }
    },
    [fetchDump],
  );

  // ── seed default data ─────────────────────────────────────────────────
  const seedDefault = useCallback(async () => {
    setUploading(true);
    setUploadStatus(null);
    try {
      await api.jdGen.seedDefault();
      setUploadStatus({ kind: "ok", counts: { employees: 100 }, mode: "seed" });
      await fetchDump();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setUploadStatus({ kind: "err", message: msg });
    } finally {
      setUploading(false);
    }
  }, [fetchDump]);

  // ── halo helper ───────────────────────────────────────────────────────
  const pulseHalo = useCallback((ids: string[]) => {
    if (!ids || ids.length === 0) return;
    setHighlighted((prev) => {
      const next = new Set(prev);
      ids.forEach((id) => next.add(id));
      return next;
    });
    ids.forEach((id) => {
      const prev = haloTimers.current.get(id);
      if (prev) window.clearTimeout(prev);
      const timer = window.setTimeout(() => {
        setHighlighted((cur) => {
          const next = new Set(cur);
          next.delete(id);
          return next;
        });
        haloTimers.current.delete(id);
      }, HALO_DURATION_MS);
      haloTimers.current.set(id, timer);
    });
  }, []);

  useEffect(
    () => () => {
      haloTimers.current.forEach((t) => window.clearTimeout(t));
      haloTimers.current.clear();
    },
    [],
  );

  // ── generate ──────────────────────────────────────────────────────────
  const runGenerate = useCallback(async () => {
    if (!roleTitle.trim()) return;
    setRunning(true);
    setToolEvents([]);
    setGenerated(null);
    setGenerationError(null);
    try {
      const stream = api.jdGen.generateStream({
        role_title: roleTitle,
        level,
        team_id: teamId.trim() || null,
      });
      for await (const ev of stream as AsyncGenerator<JDStreamEvent>) {
        if (ev.type === "tool_call") {
          setToolEvents((prev) => [
            ...prev,
            { tool: ev.tool, ok: null },
          ]);
        } else if (ev.type === "tool_result") {
          setToolEvents((prev) => {
            const next = [...prev];
            const idx = next.map((e) => e.tool).lastIndexOf(ev.tool);
            if (idx >= 0) {
              next[idx] = { tool: ev.tool, ok: ev.ok, summary: ev.summary };
            } else {
              next.push({ tool: ev.tool, ok: ev.ok, summary: ev.summary });
            }
            return next;
          });
          pulseHalo(ev.node_ids ?? []);
        } else if (ev.type === "final_full") {
          setGenerated(ev.result);
          // Refresh the graph so the new JD node appears
          fetchDump();
        } else if (ev.type === "error") {
          setGenerationError(ev.error);
        }
      }
    } catch (e) {
      setGenerationError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }, [roleTitle, level, teamId, pulseHalo, fetchDump]);

  // ── approve / reject ──────────────────────────────────────────────────
  const onApprove = useCallback(async () => {
    if (!generated) return;
    try {
      await api.jdGen.approve(generated.jd_id);
      setGenerated((g) => (g ? { ...g } : g));
    } catch (e) {
      setGenerationError(e instanceof Error ? e.message : String(e));
    }
  }, [generated]);

  const onReject = useCallback(async () => {
    if (!generated || !rejectText.trim()) return;
    setRejectBusy(true);
    try {
      await api.jdGen.reject(generated.jd_id, {
        reason_text: rejectText,
        categories: rejectCats.length ? rejectCats : undefined,
      });
      setRejectOpen(false);
      setRejectText("");
      setRejectCats([]);
      // Run generation again so the recruiter can see the rejection consumed.
      runGenerate();
    } catch (e) {
      setGenerationError(e instanceof Error ? e.message : String(e));
    } finally {
      setRejectBusy(false);
    }
  }, [generated, rejectText, rejectCats, runGenerate]);

  // ── memo: node accessors for force-graph-3d ───────────────────────────
  const graphData = useMemo(
    () =>
      dump
        ? {
            nodes: dump.nodes.map((n) => ({ ...n, _halo: highlighted.has(n.id) })),
            links: dump.links,
          }
        : { nodes: [], links: [] },
    [dump, highlighted],
  );

  // ── render ────────────────────────────────────────────────────────────
  return (
    <div className="flex h-[calc(100vh-10rem)] flex-col gap-3">
      {/* ── Top bar: data source ────────────────────────────────────── */}
      <DataSourceBar
        source={source}
        setSource={setSource}
        familyFilter={familyFilter}
        setFamilyFilter={setFamilyFilter}
        includeRejections={includeRejections}
        setIncludeRejections={setIncludeRejections}
        nodeCount={dump?.node_count ?? 0}
        linkCount={dump?.link_count ?? 0}
        uploadStatus={uploadStatus}
        uploading={uploading}
        onFileChosen={onFileChosen}
        onSeedDefault={seedDefault}
        onReload={() => fetchDump()}
      />

      {/* ── Main: 3D canvas + side drawer ─────────────────────────── */}
      <div className="flex min-h-0 flex-1 gap-3">
        {/* Canvas */}
        <div className="relative flex-1 overflow-hidden rounded-2xl border border-border bg-card/40">
          {graphLoading && <CanvasOverlay text="Loading graph…" />}
          {graphError && !graphLoading && (
            <CanvasOverlay text={graphError} tone="error" />
          )}
          {!graphError && dump && (
            <Suspense fallback={<CanvasOverlay text="Loading 3D engine…" />}>
              <ForceGraph3D
                ref={fgRef}
                graphData={graphData}
                backgroundColor="rgba(0,0,0,0)"
                nodeLabel={(n: any) => `${n.label} (${n.type})`}
                nodeColor={(n: any) =>
                  n._halo ? HALO_COLOR : NODE_COLOR[n.type as keyof typeof NODE_COLOR] ?? "#888"
                }
                nodeRelSize={5}
                linkColor={() => "rgba(148, 163, 184, 0.25)"}
                linkWidth={(l: any) =>
                  typeof l.proficiency === "number"
                    ? Math.max(0.3, l.proficiency / 2)
                    : typeof l.frequency === "number"
                    ? Math.max(0.3, Math.log10(l.frequency + 1) + 0.5)
                    : 0.4
                }
                linkOpacity={0.5}
                nodeOpacity={0.95}
                onNodeClick={(n: any) => setSelected(n as JDGraphNode)}
                cooldownTicks={reducedMotion ? 0 : 200}
                enableNodeDrag={!reducedMotion}
                nodeThreeObject={(node: any) => {
                  const geo = new THREE.SphereGeometry(node._halo ? 7 : 5);
                  const mat = new THREE.MeshLambertMaterial({
                    color:
                      node._halo
                        ? HALO_COLOR
                        : NODE_COLOR[node.type as keyof typeof NODE_COLOR] ?? "#888",
                    emissive: node._halo ? HALO_COLOR : "#000",
                    emissiveIntensity: node._halo ? 0.45 : 0.0,
                    transparent: true,
                    opacity: node._halo ? 1.0 : 0.95,
                  });
                  return new THREE.Mesh(geo, mat);
                }}
              />
            </Suspense>
          )}
          {selected && (
            <NodeDetail node={selected} onClose={() => setSelected(null)} />
          )}
          <Legend />
        </div>

        {/* Drawer */}
        <AgentDrawer
          open={drawerOpen}
          setOpen={setDrawerOpen}
          roleTitle={roleTitle}
          setRoleTitle={setRoleTitle}
          level={level}
          setLevel={setLevel}
          teamId={teamId}
          setTeamId={setTeamId}
          running={running}
          toolEvents={toolEvents}
          generated={generated}
          generationError={generationError}
          onRun={runGenerate}
          onApprove={onApprove}
          onOpenReject={() => setRejectOpen(true)}
        />
      </div>

      {/* ── Reject modal ────────────────────────────────────────────── */}
      <AnimatePresence>
        {rejectOpen && (
          <RejectModal
            text={rejectText}
            setText={setRejectText}
            cats={rejectCats}
            setCats={setRejectCats}
            busy={rejectBusy}
            onCancel={() => setRejectOpen(false)}
            onSubmit={onReject}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

// ===========================================================================
// Sub-components
// ===========================================================================
function DataSourceBar(props: {
  source: "default" | "custom";
  setSource: (s: "default" | "custom") => void;
  familyFilter: string | "all";
  setFamilyFilter: (s: string | "all") => void;
  includeRejections: boolean;
  setIncludeRejections: (v: boolean) => void;
  nodeCount: number;
  linkCount: number;
  uploadStatus:
    | null
    | { kind: "ok"; counts: Record<string, number>; mode: string }
    | { kind: "err"; message: string };
  uploading: boolean;
  onFileChosen: (f: File, mode: "augment" | "replace") => void;
  onSeedDefault: () => void;
  onReload: () => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [pendingMode, setPendingMode] = useState<"augment" | "replace">("augment");

  const FAMILIES: Array<{ id: string | "all"; label: string }> = [
    { id: "all",          label: "All families" },
    { id: "backend",      label: "Backend" },
    { id: "frontend",     label: "Frontend" },
    { id: "data_science", label: "Data Science" },
    { id: "product",      label: "Product" },
    { id: "design",       label: "Design" },
  ];

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-2xl border border-border bg-card/60 px-4 py-3 backdrop-blur-xl">
      {/* Source toggle */}
      <div
        role="tablist"
        aria-label="Data source"
        className="inline-flex rounded-lg border border-border bg-muted/40 p-1"
      >
        {(["default", "custom"] as const).map((s) => (
          <button
            key={s}
            role="tab"
            aria-selected={props.source === s}
            onClick={() => props.setSource(s)}
            className={`min-h-[36px] rounded-md px-3 text-xs font-medium transition ${
              props.source === s
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {s === "default" ? "Default data" : "Custom data"}
          </button>
        ))}
      </div>

      {/* Family filter */}
      <select
        aria-label="Filter by role family"
        value={props.familyFilter}
        onChange={(e) => props.setFamilyFilter(e.target.value as any)}
        className="min-h-[36px] rounded-lg border border-border bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        {FAMILIES.map((f) => (
          <option key={f.id} value={f.id}>
            {f.label}
          </option>
        ))}
      </select>

      {/* Include rejections */}
      <label className="inline-flex select-none items-center gap-2 text-xs text-muted-foreground">
        <input
          type="checkbox"
          checked={props.includeRejections}
          onChange={(e) => props.setIncludeRejections(e.target.checked)}
          className="h-4 w-4 rounded border-border"
        />
        Include rejections
      </label>

      {/* Counts */}
      <div className="ml-2 text-xs text-muted-foreground">
        {props.nodeCount.toLocaleString()} nodes · {props.linkCount.toLocaleString()} links
      </div>

      <div className="ml-auto flex flex-wrap items-center gap-2">
        {props.source === "default" ? (
          <Button
            variant="outline"
            size="sm"
            onClick={props.onSeedDefault}
            disabled={props.uploading}
          >
            <Database className="mr-1.5 h-3.5 w-3.5" />
            {props.uploading ? "Seeding…" : "Reseed default"}
          </Button>
        ) : (
          <>
            <select
              aria-label="Upload mode"
              value={pendingMode}
              onChange={(e) => setPendingMode(e.target.value as any)}
              className="min-h-[36px] rounded-lg border border-border bg-background px-2 text-xs"
            >
              <option value="augment">Augment</option>
              <option value="replace">Replace</option>
            </select>
            <input
              ref={fileRef}
              type="file"
              accept=".json,.zip"
              hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) props.onFileChosen(f, pendingMode);
                e.target.value = "";
              }}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={() => fileRef.current?.click()}
              disabled={props.uploading}
            >
              <FileUp className="mr-1.5 h-3.5 w-3.5" />
              {props.uploading ? "Uploading…" : "Choose file"}
            </Button>
          </>
        )}
        <Button variant="ghost" size="sm" onClick={props.onReload} title="Reload graph">
          <RotateCcw className="h-3.5 w-3.5" />
        </Button>
      </div>

      {/* Upload feedback */}
      <AnimatePresence>
        {props.uploadStatus && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            className={`w-full text-xs ${
              props.uploadStatus.kind === "ok" ? "text-emerald-500" : "text-rose-500"
            }`}
          >
            {props.uploadStatus.kind === "ok" ? (
              <>
                Done ({props.uploadStatus.mode}) ·{" "}
                {Object.entries(props.uploadStatus.counts)
                  .map(([k, v]) => `${v} ${k}`)
                  .join(" · ")}
              </>
            ) : (
              <>Upload failed: {props.uploadStatus.message}</>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function AgentDrawer(props: {
  open: boolean;
  setOpen: (v: boolean) => void;
  roleTitle: string;
  setRoleTitle: (v: string) => void;
  level: JDLevel;
  setLevel: (v: JDLevel) => void;
  teamId: string;
  setTeamId: (v: string) => void;
  running: boolean;
  toolEvents: Array<{ tool: string; ok: boolean | null; summary?: string }>;
  generated: JDGenerateResult | null;
  generationError: string | null;
  onRun: () => void;
  onApprove: () => void;
  onOpenReject: () => void;
}) {
  return (
    <motion.aside
      animate={{ width: props.open ? 380 : 44 }}
      transition={{ type: "spring", stiffness: 320, damping: 32 }}
      className="flex shrink-0 flex-col overflow-hidden rounded-2xl border border-border bg-card/60 backdrop-blur-xl"
    >
      <div className="flex items-center justify-between border-b border-border px-3 py-2">
        <button
          onClick={() => props.setOpen(!props.open)}
          aria-label={props.open ? "Collapse agent panel" : "Expand agent panel"}
          aria-expanded={props.open}
          className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          {props.open ? <ChevronRight className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </button>
        {props.open && (
          <h3 className="flex items-center gap-1.5 text-sm font-semibold">
            <Sparkles className="h-4 w-4 text-accent" aria-hidden />
            Live agent
          </h3>
        )}
        <div className="w-8" />
      </div>

      {props.open && (
        <div className="flex flex-col gap-3 overflow-y-auto p-3 text-sm">
          {/* Inputs */}
          <label className="flex flex-col gap-1">
            <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Role title
            </span>
            <input
              value={props.roleTitle}
              onChange={(e) => props.setRoleTitle(e.target.value)}
              className="min-h-[40px] rounded-lg border border-border bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              placeholder="Senior Backend Engineer"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Level
            </span>
            <select
              value={props.level}
              onChange={(e) => props.setLevel(e.target.value as JDLevel)}
              className="min-h-[40px] rounded-lg border border-border bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              <option value="junior">Junior</option>
              <option value="mid">Mid</option>
              <option value="senior">Senior</option>
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              Team (optional)
            </span>
            <input
              value={props.teamId}
              onChange={(e) => props.setTeamId(e.target.value)}
              placeholder="team_platform"
              className="min-h-[40px] rounded-lg border border-border bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            />
          </label>

          <Button
            onClick={props.onRun}
            disabled={props.running || !props.roleTitle.trim()}
            className="min-h-[44px]"
          >
            {props.running ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />
                Generating…
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" aria-hidden />
                Generate
              </>
            )}
          </Button>

          {/* Tool-call log */}
          {props.toolEvents.length > 0 && (
            <section className="flex flex-col gap-1">
              <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                Tool calls
              </div>
              <ul className="space-y-1 rounded-lg border border-border bg-muted/30 p-2">
                {props.toolEvents.map((ev, i) => (
                  <li key={`${ev.tool}-${i}`} className="flex items-start gap-2 text-xs">
                    {ev.ok === null ? (
                      <Loader2 className="mt-0.5 h-3.5 w-3.5 shrink-0 animate-spin text-muted-foreground" />
                    ) : ev.ok ? (
                      <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-500" />
                    ) : (
                      <XCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-rose-500" />
                    )}
                    <div className="min-w-0 flex-1">
                      <span className="font-mono text-foreground">{ev.tool}</span>
                      {ev.summary && (
                        <div className="mt-0.5 truncate text-muted-foreground">{ev.summary}</div>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          )}

          {/* Error */}
          {props.generationError && (
            <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-2 text-xs text-rose-500">
              {props.generationError}
            </div>
          )}

          {/* Final JD */}
          {props.generated && (
            <section className="flex flex-col gap-2">
              <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                Draft
              </div>
              <div className="rounded-lg border border-border bg-background p-3">
                <p className="whitespace-pre-wrap text-sm leading-relaxed">{props.generated.jd_text}</p>
                {props.generated.must_have.length > 0 && (
                  <div className="mt-3">
                    <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                      Must have
                    </div>
                    <ul className="mt-1 list-disc pl-4 text-xs">
                      {props.generated.must_have.map((m, i) => (
                        <li key={i}>{m}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {props.generated.nice_to_have.length > 0 && (
                  <div className="mt-2">
                    <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                      Nice to have
                    </div>
                    <ul className="mt-1 list-disc pl-4 text-xs">
                      {props.generated.nice_to_have.map((m, i) => (
                        <li key={i}>{m}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {props.generated.bias_warnings.length > 0 && (
                  <div className="mt-2 rounded border border-amber-500/30 bg-amber-500/10 p-2 text-[11px] text-amber-600">
                    <div className="font-semibold">Bias warnings</div>
                    <ul className="mt-1 list-disc pl-4">
                      {props.generated.bias_warnings.slice(0, 5).map((w, i) => (
                        <li key={i}>
                          [{w.category}] {w.term}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {/* Rejection-history mini-list */}
              {props.generated.rejection_checks.length > 0 && (
                <RejectionChecksList checks={props.generated.rejection_checks} />
              )}

              <div className="flex gap-2">
                <Button onClick={props.onApprove} variant="default" className="flex-1 min-h-[40px]">
                  <CheckCircle2 className="mr-1.5 h-4 w-4" aria-hidden />
                  Approve
                </Button>
                <Button
                  onClick={props.onOpenReject}
                  variant="outline"
                  className="flex-1 min-h-[40px]"
                >
                  <CircleSlash className="mr-1.5 h-4 w-4" aria-hidden />
                  Reject
                </Button>
              </div>
            </section>
          )}
        </div>
      )}
    </motion.aside>
  );
}

function RejectionChecksList({ checks }: { checks: JDRejectionCheck[] }) {
  return (
    <div>
      <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        Past rejections addressed
      </div>
      <ul className="mt-1 space-y-1.5 rounded-lg border border-border bg-muted/30 p-2 text-xs">
        {checks.map((c, i) => (
          <li key={i}>
            <div className="font-medium text-foreground">{c.past_reason}</div>
            <div className="mt-0.5 text-muted-foreground">→ {c.how_addressed}</div>
            {c.categories.length > 0 && (
              <div className="mt-0.5 flex flex-wrap gap-1">
                {c.categories.map((cat) => (
                  <span
                    key={cat}
                    className="rounded-full bg-accent/15 px-2 py-0.5 text-[10px] uppercase tracking-wider text-accent"
                  >
                    {cat}
                  </span>
                ))}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

const REJECT_CATEGORIES = [
  "tone", "requirements", "bias", "culture-fit", "accuracy", "structure", "other",
];

function RejectModal(props: {
  text: string;
  setText: (v: string) => void;
  cats: string[];
  setCats: (v: string[]) => void;
  busy: boolean;
  onCancel: () => void;
  onSubmit: () => void;
}) {
  return (
    <motion.div
      role="dialog"
      aria-modal="true"
      aria-label="Reject job description"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 grid place-items-center bg-foreground/40 p-4 backdrop-blur-sm"
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className="w-full max-w-md rounded-2xl border border-border bg-background p-5 shadow-2xl"
      >
        <div className="flex items-start justify-between">
          <h3 className="text-lg font-semibold">Reject draft</h3>
          <button
            onClick={props.onCancel}
            aria-label="Close"
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <label className="mt-3 block text-sm">
          <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Why? (be specific — this drives the next generation)
          </span>
          <textarea
            value={props.text}
            onChange={(e) => props.setText(e.target.value)}
            rows={4}
            placeholder="Too jargon-heavy — phrases like 'synergize verticals' don't match our voice."
            className="mt-1 w-full rounded-lg border border-border bg-background p-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          />
        </label>
        <div className="mt-3">
          <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Categories (optional)
          </div>
          <div className="mt-1 flex flex-wrap gap-1.5">
            {REJECT_CATEGORIES.map((c) => (
              <button
                key={c}
                onClick={() =>
                  props.setCats(
                    props.cats.includes(c)
                      ? props.cats.filter((x) => x !== c)
                      : [...props.cats, c],
                  )
                }
                className={`rounded-full border px-2.5 py-1 text-xs transition ${
                  props.cats.includes(c)
                    ? "border-accent bg-accent/15 text-accent"
                    : "border-border text-muted-foreground hover:text-foreground"
                }`}
                aria-pressed={props.cats.includes(c)}
              >
                {c}
              </button>
            ))}
          </div>
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <Button variant="outline" onClick={props.onCancel} disabled={props.busy}>
            Cancel
          </Button>
          <Button
            onClick={props.onSubmit}
            disabled={!props.text.trim() || props.busy}
          >
            {props.busy ? (
              <>
                <Loader2 className="mr-1.5 h-4 w-4 animate-spin" aria-hidden /> Saving…
              </>
            ) : (
              "Save & regenerate"
            )}
          </Button>
        </div>
      </motion.div>
    </motion.div>
  );
}

function NodeDetail({ node, onClose }: { node: JDGraphNode; onClose: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: 10 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 10 }}
      className="pointer-events-auto absolute right-3 top-3 max-h-[80%] w-72 overflow-y-auto rounded-xl border border-border bg-background/95 p-3 shadow-xl backdrop-blur-xl"
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
            {node.type}
          </div>
          <div className="text-sm font-semibold">{node.label}</div>
        </div>
        <button
          onClick={onClose}
          aria-label="Close detail"
          className="inline-flex h-6 w-6 items-center justify-center rounded text-muted-foreground hover:text-foreground"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      {node.props && Object.keys(node.props).length > 0 && (
        <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 text-xs">
          {Object.entries(node.props).map(([k, v]) => (
            <Suspense key={k} fallback={null}>
              <dt className="text-muted-foreground">{k}</dt>
              <dd className="break-words text-foreground">
                {typeof v === "object" ? JSON.stringify(v) : String(v ?? "—")}
              </dd>
            </Suspense>
          ))}
        </dl>
      )}
    </motion.div>
  );
}

function Legend() {
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

function CanvasOverlay({ text, tone = "default" }: { text: string; tone?: "default" | "error" }) {
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
