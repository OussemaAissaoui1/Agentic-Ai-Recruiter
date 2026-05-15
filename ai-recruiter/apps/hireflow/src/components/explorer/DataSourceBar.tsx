import { AnimatePresence, motion } from "framer-motion";
import { Database, FileUp, RotateCcw } from "lucide-react";
import { useRef, useState } from "react";

import { Button } from "@/components/ui/button";

import type { UploadStatus } from "./helpers";

const FAMILIES: Array<{ id: string | "all"; label: string }> = [
  { id: "all",          label: "All families" },
  { id: "backend",      label: "Backend" },
  { id: "frontend",     label: "Frontend" },
  { id: "data_science", label: "Data Science" },
  { id: "product",      label: "Product" },
  { id: "design",       label: "Design" },
];

export function DataSourceBar(props: {
  source: "default" | "custom";
  setSource: (s: "default" | "custom") => void;
  familyFilter: string | "all";
  setFamilyFilter: (s: string | "all") => void;
  includeRejections: boolean;
  setIncludeRejections: (v: boolean) => void;
  uploadStatus: UploadStatus;
  uploading: boolean;
  onFileChosen: (f: File, mode: "augment" | "replace") => void;
  onSeedDefault: () => void;
  onReload: () => void;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [pendingMode, setPendingMode] = useState<"augment" | "replace">("augment");

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-2xl border border-border bg-card/60 px-4 py-2.5 backdrop-blur-xl">
      {/* Group 1: data source + filter */}
      <div className="flex flex-wrap items-center gap-3">
        <div
          role="tablist"
          aria-label="Data source"
          className="inline-flex rounded-lg border border-border bg-muted/40 p-1"
        >
          {(["default", "custom"] as const).map((s) => (
            <button
              key={s}
              type="button"
              role="tab"
              aria-selected={props.source === s}
              onClick={() => props.setSource(s)}
              className={`min-h-[36px] rounded-md px-3 text-xs font-medium transition ${
                props.source === s
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {s === "default" ? "Default" : "Custom"}
            </button>
          ))}
        </div>

        <label className="inline-flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
          Family
          <select
            aria-label="Filter by role family"
            value={props.familyFilter}
            onChange={(e) => props.setFamilyFilter(e.target.value as string | "all")}
            className="min-h-[36px] rounded-md border border-border bg-background px-2 text-sm font-normal normal-case tracking-normal text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          >
            {FAMILIES.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
        </label>

        <label className="inline-flex select-none items-center gap-2 text-xs text-muted-foreground">
          <input
            type="checkbox"
            checked={props.includeRejections}
            onChange={(e) => props.setIncludeRejections(e.target.checked)}
            className="h-4 w-4 rounded border-border"
          />
          Include rejections
        </label>
      </div>

      {/* Group 2: actions */}
      <div className="ml-auto flex flex-wrap items-center gap-2">
        {props.source === "default" ? (
          <Button
            variant="outline"
            size="sm"
            onClick={props.onSeedDefault}
            disabled={props.uploading}
            className="min-h-[44px]"
          >
            <Database className="mr-1.5 h-3.5 w-3.5" aria-hidden />
            {props.uploading ? "Seeding…" : "Reseed default"}
          </Button>
        ) : (
          <>
            <select
              aria-label="Upload mode"
              value={pendingMode}
              onChange={(e) => setPendingMode(e.target.value as "augment" | "replace")}
              className="min-h-[44px] rounded-lg border border-border bg-background px-2 text-xs"
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
              className="min-h-[44px]"
            >
              <FileUp className="mr-1.5 h-3.5 w-3.5" aria-hidden />
              {props.uploading ? "Uploading…" : "Choose file"}
            </Button>
          </>
        )}
        <button
          type="button"
          onClick={props.onReload}
          aria-label="Reload graph"
          title="Reload graph"
          className="inline-flex h-11 w-11 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          <RotateCcw className="h-3.5 w-3.5" aria-hidden />
        </button>
      </div>

      {/* Upload feedback */}
      <AnimatePresence>
        {props.uploadStatus && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            role={props.uploadStatus.kind === "err" ? "alert" : "status"}
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
