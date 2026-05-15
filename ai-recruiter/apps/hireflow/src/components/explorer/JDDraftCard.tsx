import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  AlertTriangle,
  BookOpen,
  Check,
  ChevronDown,
  ChevronUp,
  Clock,
  Copy,
  Hash,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { JDGenerateResult, JDLevel } from "@/lib/api";

import {
  countWords,
  jdToMarkdown,
  readMinutesFor,
  writeToClipboard,
  type JDBiasWarning,
} from "./helpers";
import { useFocusTrap } from "./useFocusTrap";

// ===========================================================================
// JDDraftCard — the compact glance view shown inside the agent drawer.
// Header (role/level/team) → optional bias banner → body → requirement chips
// → footer (word count, read time, "Read in full" CTA, jd id).
// ===========================================================================
export function JDDraftCard({
  result,
  roleTitle,
  level,
  teamId,
}: {
  result: JDGenerateResult;
  roleTitle: string;
  level: JDLevel;
  teamId: string;
}) {
  const wordCount = useMemo(() => countWords(result.jd_text), [result.jd_text]);
  const readMinutes = readMinutesFor(wordCount);
  const team = (result.team_id || teamId || "").trim();
  const markdown = useMemo(
    () => jdToMarkdown(result, roleTitle, level, team),
    [result, roleTitle, level, team],
  );

  const [readingOpen, setReadingOpen] = useState(false);

  return (
    <article className="overflow-hidden rounded-xl border border-border bg-background shadow-sm">
      {/* Header */}
      <header className="relative border-b border-border bg-gradient-to-br from-accent/10 via-background to-background px-4 pb-3 pt-4">
        <div className="absolute left-0 top-0 h-full w-1 bg-accent" aria-hidden />
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-accent">
              Draft job description
            </div>
            <h4 className="mt-1 break-words text-base font-semibold leading-tight text-foreground">
              {roleTitle || "Untitled role"}
            </h4>
            <div className="mt-2 flex flex-wrap items-center gap-1.5">
              <MetaChip label={level} kind="level" />
              {team && <MetaChip label={team} kind="team" />}
              <MetaChip label={result.role_family.replace("_", " ")} kind="family" />
            </div>
          </div>
          <CopyMarkdownButton payload={markdown} />
        </div>
      </header>

      {/* Reading-view modal — lazily rendered via AnimatePresence */}
      <AnimatePresence>
        {readingOpen && (
          <JDReadingView
            result={result}
            roleTitle={roleTitle}
            level={level}
            team={team}
            wordCount={wordCount}
            readMinutes={readMinutes}
            markdown={markdown}
            onClose={() => setReadingOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Bias warnings banner */}
      {result.bias_warnings.length > 0 && (
        <BiasWarningsBanner warnings={result.bias_warnings} />
      )}

      {/* Body */}
      <div className="px-4 py-4">
        <p className="whitespace-pre-wrap text-[14px] leading-[1.7] text-foreground">
          {result.jd_text}
        </p>
      </div>

      {/* Requirement chips */}
      {(result.must_have.length > 0 || result.nice_to_have.length > 0) && (
        <div className="space-y-3 border-t border-border bg-muted/20 px-4 py-3">
          {result.must_have.length > 0 && (
            <RequirementGroup label="Must have" items={result.must_have} tone="primary" />
          )}
          {result.nice_to_have.length > 0 && (
            <RequirementGroup label="Nice to have" items={result.nice_to_have} tone="muted" />
          )}
        </div>
      )}

      {/* Footer */}
      <footer className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-border bg-muted/10 px-4 py-2.5 text-[11px] uppercase tracking-wider text-muted-foreground">
        <span className="inline-flex items-center gap-1">
          <Hash className="h-3 w-3" aria-hidden />
          {wordCount} words
        </span>
        <span className="inline-flex items-center gap-1">
          <Clock className="h-3 w-3" aria-hidden />
          {readMinutes} min read
        </span>
        <button
          type="button"
          onClick={() => setReadingOpen(true)}
          className="inline-flex min-h-[44px] items-center gap-1 rounded-md border border-accent/30 bg-accent/5 px-3 text-[11px] font-medium text-accent transition hover:bg-accent/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          <BookOpen className="h-3.5 w-3.5" aria-hidden />
          Read in full view
        </button>
        <span
          className="ml-auto font-mono text-[11px] normal-case tracking-normal text-muted-foreground/80"
          title={result.jd_id}
        >
          {result.jd_id.length > 12 ? `${result.jd_id.slice(0, 12)}…` : result.jd_id}
        </span>
      </footer>
    </article>
  );
}

// ===========================================================================
// MetaChip — tiny role/level/team/family tag in the card header.
// ===========================================================================
function MetaChip({
  label,
  kind,
}: {
  label: string;
  kind: "level" | "team" | "family";
}) {
  const styles = {
    level:  "border-accent/30 bg-accent/10 text-accent",
    team:   "border-border bg-background text-foreground",
    family: "border-border bg-muted/40 text-muted-foreground",
  }[kind];
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider ${styles}`}
    >
      {label}
    </span>
  );
}

// ===========================================================================
// RequirementGroup — chip cluster for must-have / nice-to-have lists.
// ===========================================================================
function RequirementGroup({
  label,
  items,
  tone,
}: {
  label: string;
  items: string[];
  tone: "primary" | "muted";
}) {
  const chipStyles =
    tone === "primary"
      ? "border-accent/30 bg-accent/10 text-accent"
      : "border-border bg-background text-muted-foreground";
  return (
    <div>
      <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        <span>{label}</span>
        <span className="rounded-full bg-muted px-1.5 py-px text-[9px] tabular-nums text-muted-foreground">
          {items.length}
        </span>
      </div>
      <div className="mt-1.5 flex flex-wrap gap-1.5">
        {items.map((item, i) => (
          <span
            key={i}
            className={`inline-flex items-center rounded-md border px-2 py-1 text-[12px] leading-tight ${chipStyles}`}
          >
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

// ===========================================================================
// BiasWarningsBanner — collapsible amber summary above the JD body.
// ===========================================================================
function BiasWarningsBanner({ warnings }: { warnings: JDBiasWarning[] }) {
  const [expanded, setExpanded] = useState(false);
  const count = warnings.length;
  return (
    <div className="border-b border-amber-500/30 bg-amber-500/5">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        aria-expanded={expanded}
        className="flex min-h-[44px] w-full items-center justify-between gap-2 px-4 py-2 text-left transition hover:bg-amber-500/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-500/40"
      >
        <span className="flex items-center gap-2 text-xs font-medium text-amber-700 dark:text-amber-400">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden />
          {count} potentially biased term{count === 1 ? "" : "s"} flagged
        </span>
        {expanded ? (
          <ChevronUp className="h-3.5 w-3.5 text-amber-700 dark:text-amber-400" aria-hidden />
        ) : (
          <ChevronDown className="h-3.5 w-3.5 text-amber-700 dark:text-amber-400" aria-hidden />
        )}
      </button>
      {expanded && (
        <ul className="space-y-1 px-4 pb-3 pt-1 text-xs">
          {warnings.slice(0, 8).map((w, i) => (
            <li
              key={i}
              className="flex items-baseline gap-2 text-amber-700 dark:text-amber-400/90"
            >
              <span className="rounded bg-amber-500/15 px-1.5 py-0.5 font-mono text-[11px]">
                {w.term}
              </span>
              <span className="text-[11px] text-amber-700/80 dark:text-amber-400/70">
                {w.category}
              </span>
            </li>
          ))}
          {warnings.length > 8 && (
            <li className="text-[11px] italic text-amber-700/60 dark:text-amber-400/50">
              + {warnings.length - 8} more…
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

// ===========================================================================
// CopyMarkdownButton — 44×44 button with copied-state feedback.
// ===========================================================================
function CopyMarkdownButton({ payload }: { payload: string }) {
  const [copied, setCopied] = useState(false);
  const onClick = useCallback(async () => {
    await writeToClipboard(payload);
    setCopied(true);
    const t = window.setTimeout(() => setCopied(false), 1800);
    return () => window.clearTimeout(t);
  }, [payload]);
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={copied ? "Copied to clipboard" : "Copy as markdown"}
      title={copied ? "Copied!" : "Copy as markdown"}
      className={`inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-md border transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
        copied
          ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-500"
          : "border-border bg-background text-muted-foreground hover:bg-muted hover:text-foreground"
      }`}
    >
      {copied ? (
        <Check className="h-4 w-4" aria-hidden />
      ) : (
        <Copy className="h-4 w-4" aria-hidden />
      )}
    </button>
  );
}

// ===========================================================================
// JDReadingView — focused reading modal.
//
// Layout:
//   < lg: single column (matches original; meta as inline dot-line above body)
//   ≥ lg: 220px sticky meta sidebar + main reading column
//
// Polish:
//   - Scroll-progress bar at the top edge of the modal (accent fill)
//   - Numbered editorial sections (01 / 02) instead of color bars
//   - Body at 18px / 1.8 line-height, 68ch measure with auto-hyphens
//   - Title at 4xl on desktop for real cover-page presence
// ===========================================================================
function JDReadingView({
  result,
  roleTitle,
  level,
  team,
  wordCount,
  readMinutes,
  markdown,
  onClose,
}: {
  result: JDGenerateResult;
  roleTitle: string;
  level: JDLevel;
  team: string;
  wordCount: number;
  readMinutes: number;
  markdown: string;
  onClose: () => void;
}) {
  const reducedMotion = useReducedMotion();
  const containerRef = useFocusTrap<HTMLDivElement>(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [progress, setProgress] = useState(0);

  // Esc to close.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Body scroll lock so the page behind doesn't drift.
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);

  // Reading-progress tracker. The progress bar at the top of the modal
  // fills from 0% to 100% as the user scrolls through the JD body.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const max = el.scrollHeight - el.clientHeight;
      setProgress(max > 0 ? Math.min(1, Math.max(0, el.scrollTop / max)) : 0);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  // Section numbering: only number if both must_have AND nice_to_have exist.
  const hasMust = result.must_have.length > 0;
  const hasNice = result.nice_to_have.length > 0;
  const mustNumber = hasMust && hasNice ? "01" : null;
  const niceNumber = hasMust && hasNice ? "02" : null;

  return (
    <motion.div
      role="dialog"
      aria-modal="true"
      aria-labelledby="jd-reading-title"
      initial={reducedMotion ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={reducedMotion ? undefined : { opacity: 0 }}
      transition={{ duration: 0.18 }}
      className="fixed inset-0 z-50 grid place-items-center bg-foreground/45 p-4 backdrop-blur-md sm:p-6"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <motion.article
        ref={containerRef}
        initial={reducedMotion ? false : { opacity: 0, y: 12, scale: 0.99 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={reducedMotion ? undefined : { opacity: 0, y: 8, scale: 0.99 }}
        transition={{ duration: 0.22, ease: "easeOut" }}
        className="relative flex max-h-[92dvh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-border bg-background shadow-2xl"
      >
        {/* Scroll progress bar — pinned to top edge */}
        <div className="absolute inset-x-0 top-0 z-20 h-0.5 bg-transparent">
          <div
            aria-hidden
            className="h-full bg-accent transition-[width] duration-100 ease-out"
            style={{ width: `${progress * 100}%` }}
          />
        </div>

        {/* Top bar */}
        <div className="flex shrink-0 items-center justify-between gap-3 border-b border-border bg-background/95 px-5 py-3 backdrop-blur-xl">
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            <BookOpen className="h-3.5 w-3.5 text-accent" aria-hidden />
            Reading view
          </div>
          <div className="flex items-center gap-2">
            <CopyMarkdownButton payload={markdown} />
            <button
              type="button"
              onClick={onClose}
              autoFocus
              aria-label="Close reading view"
              className="inline-flex h-11 w-11 items-center justify-center rounded-md border border-border bg-background text-muted-foreground transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              <X className="h-4 w-4" aria-hidden />
            </button>
          </div>
        </div>

        {/* Scrollable body */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {result.bias_warnings.length > 0 && (
            <BiasCallout warnings={result.bias_warnings} />
          )}

          <div className="grid gap-x-10 gap-y-6 px-6 py-8 sm:px-10 sm:py-10 lg:grid-cols-[220px_1fr] lg:gap-y-8">
            {/* Meta sidebar (lg+) */}
            <aside className="hidden lg:block">
              <div className="sticky top-4 flex flex-col gap-5">
                <MetaItem label="Level" value={level} capitalize />
                <MetaItem
                  label="Family"
                  value={result.role_family.replace("_", " ")}
                  capitalize
                />
                {team && <MetaItem label="Team" value={team} />}
                <hr className="border-border" />
                <MetaItem label="Read time" value={`${readMinutes} min`} />
                <MetaItem label="Words" value={wordCount.toLocaleString()} />
                <MetaItem label="Requirements" value={`${result.must_have.length + result.nice_to_have.length}`} />
                <hr className="border-border" />
                <MetaItem label="JD ID" value={result.jd_id} mono />
              </div>
            </aside>

            {/* Main reading column */}
            <div className="min-w-0">
              <header className="border-b border-border pb-6">
                <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-accent">
                  Job description
                </div>
                <h2
                  id="jd-reading-title"
                  className="mt-2.5 text-[2.25rem] font-semibold leading-[1.05] tracking-tight text-foreground sm:text-[2.75rem] lg:text-[3rem]"
                >
                  {roleTitle || "Untitled role"}
                </h2>
                {/* Inline meta — hidden on lg+ (sidebar replaces it) */}
                <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted-foreground lg:hidden">
                  <span className="capitalize">{level}</span>
                  <span aria-hidden className="text-muted-foreground/40">·</span>
                  <span className="capitalize">{result.role_family.replace("_", " ")}</span>
                  {team && (
                    <>
                      <span aria-hidden className="text-muted-foreground/40">·</span>
                      <span>{team}</span>
                    </>
                  )}
                  <span aria-hidden className="text-muted-foreground/40">·</span>
                  <span className="inline-flex items-center gap-1">
                    <Clock className="h-3.5 w-3.5" aria-hidden />
                    {readMinutes} min read
                  </span>
                </div>
              </header>

              <div
                className="mt-8 whitespace-pre-wrap text-[18px] leading-[1.8] text-foreground"
                style={{ maxWidth: "68ch", hyphens: "auto" }}
              >
                {result.jd_text}
              </div>

              {hasMust && (
                <NumberedSection
                  number={mustNumber}
                  title="Must have"
                  items={result.must_have}
                  tone="primary"
                />
              )}
              {hasNice && (
                <NumberedSection
                  number={niceNumber}
                  title="Nice to have"
                  items={result.nice_to_have}
                  tone="muted"
                />
              )}
            </div>
          </div>
        </div>

        {/* Minimal footer — meta lives in the sidebar on lg+, in the header on smaller */}
        <footer className="flex shrink-0 items-center justify-between gap-3 border-t border-border bg-muted/10 px-6 py-3 text-[11px] text-muted-foreground sm:px-10">
          <span className="inline-flex items-center gap-1.5">
            <Hash className="h-3 w-3" aria-hidden />
            {wordCount} words · {readMinutes} min read
          </span>
          <span className="font-mono text-[11px] text-muted-foreground/70">
            Generated by Graph-RAG agent
          </span>
        </footer>
      </motion.article>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// MetaItem — vertical label/value pair used inside the reading-view sidebar.
// ---------------------------------------------------------------------------
function MetaItem({
  label,
  value,
  mono,
  capitalize,
}: {
  label: string;
  value: string;
  mono?: boolean;
  capitalize?: boolean;
}) {
  return (
    <div>
      <dt className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </dt>
      <dd
        className={`mt-1 ${mono ? "break-all font-mono text-[11px] text-foreground/80" : "text-sm text-foreground"} ${capitalize ? "capitalize" : ""}`}
      >
        {value}
      </dd>
    </div>
  );
}

// ---------------------------------------------------------------------------
// NumberedSection — editorial-style section with a small index number,
// horizontal rule, and bulleted list of requirements.
// ---------------------------------------------------------------------------
function NumberedSection({
  number,
  title,
  items,
  tone,
}: {
  number: string | null;
  title: string;
  items: string[];
  tone: "primary" | "muted";
}) {
  const dotColor = tone === "primary" ? "bg-accent" : "bg-muted-foreground/40";
  const ruleColor = tone === "primary" ? "bg-accent/50" : "bg-border";
  return (
    <section className="mt-14">
      <div className="flex items-baseline gap-4">
        {number && (
          <span className="font-mono text-xs font-semibold tracking-[0.16em] text-muted-foreground">
            {number}
          </span>
        )}
        <span aria-hidden className={`h-px flex-1 ${ruleColor}`} />
        <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] tabular-nums text-muted-foreground">
          {items.length}
        </span>
      </div>
      <h3 className="mt-3 text-xl font-semibold tracking-tight text-foreground sm:text-2xl">
        {title}
      </h3>
      <ul className="mt-5 space-y-3 text-[16px] leading-[1.7] text-foreground">
        {items.map((item, i) => (
          <li key={i} className="flex gap-3">
            <span
              aria-hidden
              className={`mt-2.5 h-1.5 w-1.5 shrink-0 rounded-full ${dotColor}`}
            />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

// ---------------------------------------------------------------------------
// BiasCallout — full-width amber alert with chips below.
// ---------------------------------------------------------------------------
function BiasCallout({ warnings }: { warnings: JDBiasWarning[] }) {
  return (
    <div className="border-b border-amber-500/30 bg-amber-500/[0.06] px-6 py-4 sm:px-10">
      <div className="flex items-start gap-3">
        <div className="grid h-8 w-8 shrink-0 place-items-center rounded-md bg-amber-500/15 text-amber-700 dark:text-amber-400">
          <AlertTriangle className="h-4 w-4" aria-hidden />
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-amber-700 dark:text-amber-400">
            {warnings.length} potentially biased term{warnings.length === 1 ? "" : "s"} flagged
          </h3>
          <p className="mt-0.5 text-xs leading-relaxed text-amber-700/80 dark:text-amber-400/80">
            Review and consider rephrasing before posting — these terms are
            statistically associated with reduced applicant diversity.
          </p>
          <ul className="mt-3 flex flex-wrap gap-1.5">
            {warnings.slice(0, 12).map((w, i) => (
              <li
                key={i}
                className="inline-flex items-center gap-1.5 rounded-md border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-xs text-amber-700 dark:text-amber-400"
              >
                <span className="font-mono">{w.term}</span>
                <span className="text-[10px] uppercase tracking-wider text-amber-700/70 dark:text-amber-400/70">
                  {w.category}
                </span>
              </li>
            ))}
            {warnings.length > 12 && (
              <li className="self-center text-[11px] italic text-amber-700/60 dark:text-amber-400/50">
                + {warnings.length - 12} more…
              </li>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
}
