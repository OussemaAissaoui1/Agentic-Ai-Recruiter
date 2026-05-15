// Knowledge graph + JD generation studio.
//
// Layout: hero header on top, then a two-column studio — ComposePanel (form +
// approve actions) on the left, ExplorerStage (tabbed Graph / Draft / Activity)
// on the right. The thin DataSourceBar lives between the header and the
// workspace.

import { createFileRoute } from "@tanstack/react-router";
import { AnimatePresence, useReducedMotion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ComposePanel } from "@/components/explorer/ComposePanel";
import { DataSourceBar } from "@/components/explorer/DataSourceBar";
import { ExplorerHeader } from "@/components/explorer/ExplorerHeader";
import { ExplorerStage } from "@/components/explorer/ExplorerStage";
import {
  isAbortError,
  type ApprovalState,
  type ToolEvent,
  type UploadStatus,
} from "@/components/explorer/helpers";
import { RejectModal } from "@/components/explorer/RejectModal";
import { HALO_DURATION_MS } from "@/components/explorer/three-cache";
import {
  api,
  type JDGenerateResult,
  type JDGraphDump,
  type JDGraphNode,
  type JDLevel,
  type JDStreamEvent,
} from "@/lib/api";

export const Route = createFileRoute("/app/explorer")({
  head: () => ({ meta: [{ title: "Graph Explorer — HireFlow" }] }),
  component: Explorer,
});

function Explorer() {
  const reducedMotion = useReducedMotion();

  // ── graph data ────────────────────────────────────────────────────────
  const [dump, setDump] = useState<JDGraphDump | null>(null);
  const [graphLoading, setGraphLoading] = useState(true);
  const [graphError, setGraphError] = useState<string | null>(null);

  // ── filter state ──────────────────────────────────────────────────────
  const [familyFilter, setFamilyFilter] = useState<string | "all">("all");
  const [includeRejections, setIncludeRejections] = useState(true);

  // ── selection / data source / upload ──────────────────────────────────
  const [selected, setSelected] = useState<JDGraphNode | null>(null);
  const [source, setSource] = useState<"default" | "custom">("default");
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>(null);
  const [uploading, setUploading] = useState(false);

  // ── compose / generate state ──────────────────────────────────────────
  const [roleTitle, setRoleTitle] = useState("Senior Backend Engineer");
  const [level, setLevel] = useState<JDLevel>("senior");
  const [teamId, setTeamId] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [toolEvents, setToolEvents] = useState<ToolEvent[]>([]);
  const [generated, setGenerated] = useState<JDGenerateResult | null>(null);
  const [generationError, setGenerationError] = useState<string | null>(null);

  // ── reject modal ──────────────────────────────────────────────────────
  const [rejectOpen, setRejectOpen] = useState(false);
  const [rejectText, setRejectText] = useState("");
  const [rejectCats, setRejectCats] = useState<string[]>([]);
  const [rejectBusy, setRejectBusy] = useState(false);

  // ── approval lifecycle ────────────────────────────────────────────────
  const [approval, setApproval] = useState<ApprovalState>({ kind: "idle" });

  // ── highlighted nodes (halo) ──────────────────────────────────────────
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
        if (signal?.aborted) return;
        setDump(data);
      } catch (e) {
        if (isAbortError(e, signal)) return;
        const msg = e instanceof Error ? e.message : String(e);
        if (/501/.test(msg) || /not implemented/i.test(msg)) {
          setGraphError(
            "Graph dump endpoint not enabled yet. Set up Neo4j and seed default data.",
          );
        } else {
          setGraphError(msg);
        }
      } finally {
        if (!signal?.aborted) setGraphLoading(false);
      }
    },
    [familyFilter, includeRejections],
  );

  useEffect(() => {
    const ctrl = new AbortController();
    fetchDump(ctrl.signal);
    return () => ctrl.abort(new DOMException("explorer effect cleanup", "AbortError"));
  }, [fetchDump]);

  useEffect(() => {
    if (!uploadStatus || uploadStatus.kind !== "ok") return;
    const t = window.setTimeout(() => setUploadStatus(null), 4000);
    return () => window.clearTimeout(t);
  }, [uploadStatus]);

  // ── upload + seed ─────────────────────────────────────────────────────
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
    setApproval({ kind: "idle" });
    try {
      const stream = api.jdGen.generateStream({
        role_title: roleTitle,
        level,
        team_id: teamId.trim() || null,
      });
      for await (const ev of stream as AsyncGenerator<JDStreamEvent>) {
        if (ev.type === "tool_call") {
          setToolEvents((prev) => [...prev, { tool: ev.tool, ok: null }]);
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

  // ── approve = mark in Neo4j + create recruit DB job ───────────────────
  const onApprove = useCallback(async () => {
    if (!generated || approval.kind === "posting") return;
    setApproval({ kind: "posting" });
    try {
      await api.jdGen.approve(generated.jd_id);
      const job = await api.jobs.create({
        title:        roleTitle.trim(),
        team:         teamId.trim() || undefined,
        level,
        description:  generated.jd_text,
        must_have:    generated.must_have,
        nice_to_have: generated.nice_to_have,
        status:       "open",
      });
      setApproval({ kind: "posted", jobId: job.id });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setApproval({ kind: "error", message: msg });
    }
  }, [generated, approval.kind, roleTitle, level, teamId]);

  // ── reject ────────────────────────────────────────────────────────────
  const onReject = useCallback(async () => {
    if (!generated || !rejectText.trim()) return;
    setRejectBusy(true);
    try {
      await api.jdGen.reject(generated.jd_id, {
        reason_text: rejectText,
        categories:  rejectCats.length ? rejectCats : undefined,
      });
      setRejectOpen(false);
      setRejectText("");
      setRejectCats([]);
      runGenerate();
    } catch (e) {
      setGenerationError(e instanceof Error ? e.message : String(e));
    } finally {
      setRejectBusy(false);
    }
  }, [generated, rejectText, rejectCats, runGenerate]);

  // ── derived ───────────────────────────────────────────────────────────
  const isEmpty = !graphLoading && !graphError && (!dump || dump.nodes.length === 0);
  const familyCount = useMemo(() => {
    if (!dump) return 0;
    const families = new Set<string>();
    for (const n of dump.nodes) {
      const fam = (n.props?.role_family ?? n.props?.family) as string | undefined;
      if (fam) families.add(fam);
    }
    return families.size;
  }, [dump]);

  const headerStatus: "loading" | "ready" | "error" | "empty" = graphError
    ? "error"
    : graphLoading
    ? "loading"
    : isEmpty
    ? "empty"
    : "ready";

  // ── render ────────────────────────────────────────────────────────────
  return (
    <div className="flex h-[calc(100dvh-5rem)] min-h-[600px] flex-col gap-4 pb-4">
      <ExplorerHeader
        nodeCount={dump?.node_count ?? 0}
        linkCount={dump?.link_count ?? 0}
        familyCount={familyCount}
        source={source}
        status={headerStatus}
      />

      <DataSourceBar
        source={source}
        setSource={setSource}
        familyFilter={familyFilter}
        setFamilyFilter={setFamilyFilter}
        includeRejections={includeRejections}
        setIncludeRejections={setIncludeRejections}
        uploadStatus={uploadStatus}
        uploading={uploading}
        onFileChosen={onFileChosen}
        onSeedDefault={seedDefault}
        onReload={() => fetchDump()}
      />

      {/* Studio: ComposePanel (left, 340px sticky) + ExplorerStage (right, fills) */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-4 lg:grid-cols-[340px_1fr]">
        <ComposePanel
          roleTitle={roleTitle}
          setRoleTitle={setRoleTitle}
          level={level}
          setLevel={setLevel}
          teamId={teamId}
          setTeamId={setTeamId}
          running={running}
          hasGenerated={!!generated}
          approval={approval}
          onRun={runGenerate}
          onApprove={onApprove}
          onOpenReject={() => setRejectOpen(true)}
        />

        <ExplorerStage
          dump={dump}
          graphLoading={graphLoading}
          graphError={graphError}
          isEmpty={isEmpty}
          source={source}
          uploading={uploading}
          onSeedDefault={seedDefault}
          highlighted={highlighted}
          selected={selected}
          setSelected={setSelected}
          reducedMotion={reducedMotion}
          generated={generated}
          generationError={generationError}
          toolEvents={toolEvents}
          running={running}
          roleTitle={roleTitle}
          level={level}
          teamId={teamId}
        />
      </div>

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
