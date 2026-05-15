// Pure helpers + small types used across the explorer route.
// No React imports — safe to import from anywhere.

import type { JDGenerateResult, JDLevel } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types local to the explorer
// ---------------------------------------------------------------------------
export type ApprovalState =
  | { kind: "idle" }
  | { kind: "posting" }
  | { kind: "posted"; jobId: string }
  | { kind: "error"; message: string };

export type JDBiasWarning = JDGenerateResult["bias_warnings"][number];

export type UploadStatus =
  | null
  | { kind: "ok"; counts: Record<string, number>; mode: string }
  | { kind: "err"; message: string };

export type ToolEvent = {
  tool: string;
  ok: boolean | null;
  summary?: string;
};

// ---------------------------------------------------------------------------
// AbortError detection — surface aborted requests cleanly without leaking
// "signal is aborted without reason" into the user-visible error path.
// ---------------------------------------------------------------------------
export function isAbortError(e: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted) return true;
  if (e instanceof DOMException && e.name === "AbortError") return true;
  if (e instanceof Error && e.name === "AbortError") return true;
  return false;
}

// ---------------------------------------------------------------------------
// JD content helpers
// ---------------------------------------------------------------------------
export function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

export function readMinutesFor(words: number): number {
  // 200 WPM is the conservative average reading speed for prose.
  return Math.max(1, Math.ceil(words / 200));
}

export function jdToMarkdown(
  r: JDGenerateResult,
  title: string,
  level: JDLevel,
  team: string,
): string {
  const lines: string[] = [];
  lines.push(`# ${title || "Untitled role"}`);
  const meta = [level && `**Level:** ${level}`, team && `**Team:** ${team}`].filter(Boolean);
  if (meta.length) lines.push(meta.join("  ·  "));
  lines.push("", r.jd_text, "");
  if (r.must_have.length) {
    lines.push("## Must have");
    r.must_have.forEach((m) => lines.push(`- ${m}`));
    lines.push("");
  }
  if (r.nice_to_have.length) {
    lines.push("## Nice to have");
    r.nice_to_have.forEach((m) => lines.push(`- ${m}`));
  }
  return lines.join("\n").trim() + "\n";
}

// ---------------------------------------------------------------------------
// Clipboard with execCommand fallback. Plain `navigator.clipboard.writeText`
// fails on insecure dev origins and when the page isn't focused, which is
// surprisingly common during local dev.
// ---------------------------------------------------------------------------
export async function writeToClipboard(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
    return;
  } catch {
    // fall through
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand("copy");
  } finally {
    document.body.removeChild(ta);
  }
}
