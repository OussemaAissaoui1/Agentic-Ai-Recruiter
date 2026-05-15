import { motion, useReducedMotion } from "framer-motion";
import { Loader2, X } from "lucide-react";
import { useEffect } from "react";

import { Button } from "@/components/ui/button";

import { useFocusTrap } from "./useFocusTrap";

const REJECT_CATEGORIES = [
  "tone",
  "requirements",
  "bias",
  "culture-fit",
  "accuracy",
  "structure",
  "other",
];

export function RejectModal({
  text,
  setText,
  cats,
  setCats,
  busy,
  onCancel,
  onSubmit,
}: {
  text: string;
  setText: (v: string) => void;
  cats: string[];
  setCats: (v: string[]) => void;
  busy: boolean;
  onCancel: () => void;
  onSubmit: () => void;
}) {
  const reducedMotion = useReducedMotion();
  const containerRef = useFocusTrap<HTMLDivElement>(true);

  // Esc closes.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !busy) {
        e.preventDefault();
        onCancel();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onCancel, busy]);

  // Body scroll lock.
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);

  const toggleCat = (c: string) =>
    setCats(cats.includes(c) ? cats.filter((x) => x !== c) : [...cats, c]);

  return (
    <motion.div
      role="dialog"
      aria-modal="true"
      aria-labelledby="reject-modal-title"
      initial={reducedMotion ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={reducedMotion ? undefined : { opacity: 0 }}
      transition={{ duration: 0.18 }}
      className="fixed inset-0 z-50 grid place-items-center bg-foreground/45 p-4 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget && !busy) onCancel();
      }}
    >
      <motion.div
        ref={containerRef}
        initial={reducedMotion ? false : { scale: 0.96, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={reducedMotion ? undefined : { scale: 0.96, opacity: 0 }}
        transition={{ duration: 0.18, ease: "easeOut" }}
        className="w-full max-w-md rounded-2xl border border-border bg-background p-5 shadow-2xl"
      >
        <div className="flex items-start justify-between">
          <h3 id="reject-modal-title" className="text-lg font-semibold">
            Reject draft
          </h3>
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            aria-label="Close"
            className="inline-flex h-11 w-11 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-50"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        </div>
        <label className="mt-3 block text-sm">
          <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
            Why? (be specific — this drives the next generation)
          </span>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={4}
            autoFocus
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
                type="button"
                onClick={() => toggleCat(c)}
                aria-pressed={cats.includes(c)}
                className={`min-h-[36px] rounded-full border px-3 py-1 text-xs transition ${
                  cats.includes(c)
                    ? "border-accent bg-accent/15 text-accent"
                    : "border-border text-muted-foreground hover:text-foreground"
                }`}
              >
                {c}
              </button>
            ))}
          </div>
        </div>
        <div className="mt-5 flex justify-end gap-2">
          <Button
            variant="outline"
            onClick={onCancel}
            disabled={busy}
            className="min-h-[44px]"
          >
            Cancel
          </Button>
          <Button
            onClick={onSubmit}
            disabled={!text.trim() || busy}
            className="min-h-[44px]"
          >
            {busy ? (
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
