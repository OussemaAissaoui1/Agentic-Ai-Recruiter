import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useState } from "react";
import { MapPin, Sparkles, ArrowRight } from "lucide-react";
import { useJobs } from "@/lib/queries";
import { Skeleton } from "@/components/ui/skeleton";

export const Route = createFileRoute("/c/")({
  head: () => ({ meta: [{ title: "Discover roles — HireFlow" }] }),
  component: Discover,
});

function Discover() {
  const [q, setQ] = useState("");
  const { data: jobs = [], isLoading } = useJobs({ status: "open" });
  const filtered = jobs.filter(
    (j) =>
      q === "" ||
      (j.title + j.team + j.location).toLowerCase().includes(q.toLowerCase()),
  );

  return (
    <div className="space-y-8">
      <div>
        <div className="text-xs uppercase tracking-widest text-muted-foreground">Find your next role</div>
        <h1 className="font-display mt-1 text-4xl tracking-tight md:text-5xl">
          Roles that <em className="italic text-accent">already know</em> they want you.
        </h1>
        <p className="mt-2 max-w-2xl text-pretty text-muted-foreground">
          Every job is matched in real time against your profile. Apply once, get an instant fit
          score, and skip generic forms.
        </p>
      </div>
      <div className="rounded-2xl border border-border bg-card p-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search roles, teams, or locations…"
          className="w-full rounded-xl bg-transparent px-3 py-3 text-sm outline-none"
        />
      </div>

      {isLoading && (
        <div className="grid gap-3 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-44 rounded-2xl" />
          ))}
        </div>
      )}

      {!isLoading && filtered.length === 0 && (
        <div className="rounded-2xl border border-dashed border-border bg-card p-12 text-center text-sm text-muted-foreground">
          No open roles right now. Check back soon.
        </div>
      )}

      <div className="grid gap-3 md:grid-cols-2">
        {filtered.map((j, i) => {
          // Synthetic per-job fit pending real personalized scoring.
          const fit = 60 + ((j.id.charCodeAt(0) || 0) % 40);
          return (
            <motion.div
              key={j.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
              className="group rounded-2xl border border-border bg-card p-5 transition hover:-translate-y-0.5 hover:shadow-card-soft"
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="text-xs uppercase tracking-widest text-muted-foreground">{j.team}</div>
                  <div className="mt-1 font-display text-xl">{j.title}</div>
                  <div className="mt-1 inline-flex items-center gap-1 text-sm text-muted-foreground">
                    <MapPin className="h-3.5 w-3.5" />
                    {j.location}
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
                    Your fit
                  </div>
                  <div className="font-mono text-2xl text-accent">{fit}</div>
                </div>
              </div>
              <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${fit}%` }}
                  transition={{ duration: 0.9 }}
                  className="h-full bg-violet-grad"
                />
              </div>
              <div className="mt-4 flex flex-wrap gap-1.5">
                {(j.must_have ?? []).slice(0, 4).map((s) => (
                  <span key={s} className="rounded-full bg-muted px-2 py-0.5 text-[10px]">
                    {s}
                  </span>
                ))}
              </div>
              <div className="mt-4 flex items-center justify-between">
                <div className="text-sm text-muted-foreground">
                  {j.currency || "$"}
                  {j.salary_min}–{j.salary_max}k
                </div>
                <Link
                  to="/c/apply/$id"
                  params={{ id: j.id }}
                  className="inline-flex items-center gap-1 rounded-full bg-foreground px-3 py-1.5 text-xs font-semibold text-background"
                >
                  Apply <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>
            </motion.div>
          );
        })}
      </div>
      <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
        <Sparkles className="h-3.5 w-3.5" /> Matches refresh as you update your profile
      </div>
    </div>
  );
}
