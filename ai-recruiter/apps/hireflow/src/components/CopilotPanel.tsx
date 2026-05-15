import { useApp } from "@/lib/store";
import { AnimatePresence, motion } from "framer-motion";
import { Sparkles, X, Send } from "lucide-react";
import { useEffect, useState } from "react";
import { useStreamingText } from "@/hooks/useStreamingText";

const PROMPTS = [
  "Draft a warm rejection for a strong candidate",
  "Summarize today's interview pipeline",
  "Suggest 3 outreach messages for senior backend engineers",
  "What's slowing the Platform pipeline?",
];

export function CopilotPanel() {
  const { copilotOpen, setCopilotOpen } = useApp();
  const [prompt, setPrompt] = useState("");
  const [submitted, setSubmitted] = useState<string | null>(null);

  useEffect(() => {
    if (!copilotOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setCopilotOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [copilotOpen, setCopilotOpen]);

  const reply =
    submitted &&
    `Here's a draft based on "${submitted}":\n\nHi {Name},\n\nThank you for the energy you brought to our process — your experience shipping at scale clearly stood out. After careful review, we've decided to move forward with another candidate whose background more closely matches the immediate needs of this role.\n\nThis was not an easy call. We'd love to keep in touch as new openings emerge across Platform and Growth, where your strengths would shine.\n\nWith appreciation,\n{Recruiter}`;
  const { text, done } = useStreamingText(reply || "", 14, [submitted]);

  return (
    <AnimatePresence>
      {copilotOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setCopilotOpen(false)}
            className="fixed inset-0 z-40 bg-foreground/40 backdrop-blur-sm"
            aria-hidden
          />
          <motion.aside
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 320, damping: 36 }}
            role="dialog"
            aria-modal="true"
            aria-label="AI Copilot"
            className="fixed right-0 top-0 z-50 flex h-screen w-full max-w-md flex-col border-l border-border bg-card shadow-2xl"
          >
            <header className="flex items-center justify-between border-b border-border px-5 py-4">
              <div className="flex items-center gap-2">
                <div className="grid h-8 w-8 place-items-center rounded-lg bg-violet-grad text-white" aria-hidden>
                  <Sparkles className="h-4 w-4" />
                </div>
                <div>
                  <div className="text-sm font-semibold">AI Copilot</div>
                  <div className="text-xs text-muted-foreground">Drafts, summaries, outreach</div>
                </div>
              </div>
              <button
                onClick={() => setCopilotOpen(false)}
                aria-label="Close Copilot (Esc)"
                className="flex min-h-[40px] min-w-[40px] items-center justify-center rounded-md p-1.5 hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                <X className="h-4 w-4" aria-hidden />
              </button>
            </header>

            <div className="flex-1 space-y-4 overflow-y-auto p-5">
              {!submitted && (
                <>
                  <p className="text-sm text-muted-foreground">Try a prompt</p>
                  <div className="grid gap-2">
                    {PROMPTS.map((p) => (
                      <button
                        key={p}
                        onClick={() => { setSubmitted(p); setPrompt(p); }}
                        className="rounded-xl border border-border bg-muted/40 p-3 text-left text-sm transition hover:border-accent/40 hover:bg-accent/5"
                      >
                        {p}
                      </button>
                    ))}
                  </div>
                </>
              )}
              {submitted && (
                <div className="space-y-3">
                  <div className="text-xs text-muted-foreground">You asked</div>
                  <div className="rounded-xl bg-muted/50 p-3 text-sm">{submitted}</div>
                  <div className="text-xs text-muted-foreground">Copilot</div>
                  <pre className="whitespace-pre-wrap rounded-xl border border-border bg-card p-4 text-sm leading-relaxed">
                    {text}
                    {!done && <span className="ml-0.5 inline-block h-3 w-1.5 animate-pulse bg-accent align-middle" />}
                  </pre>
                </div>
              )}
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (prompt.trim()) setSubmitted(prompt.trim());
              }}
              className="flex gap-2 border-t border-border bg-background/60 p-3"
            >
              <label htmlFor="copilot-prompt" className="sr-only">
                Ask Copilot
              </label>
              <input
                id="copilot-prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Ask anything…"
                className="min-h-[40px] flex-1 rounded-lg border border-border bg-card px-3 py-2 text-sm outline-none focus:border-accent focus-visible:ring-2 focus-visible:ring-accent"
              />
              <button
                type="submit"
                aria-label="Send to Copilot"
                disabled={!prompt.trim()}
                className="grid h-10 w-10 place-items-center rounded-lg bg-violet-grad text-white shadow-glow transition hover:opacity-90 disabled:opacity-40"
              >
                <Send className="h-4 w-4" aria-hidden />
              </button>
            </form>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
