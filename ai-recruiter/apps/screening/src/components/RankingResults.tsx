import { RankResponse } from "../api";

export function RankingResults({ response }: { response: RankResponse }) {
  const selected = new Set(response.ga?.selected_ids ?? []);
  return (
    <div className="bg-slate-800 rounded-xl p-5 shadow">
      <h2 className="text-lg font-semibold mb-1">Ranking</h2>
      <div className="text-xs text-slate-400 mb-4">
        model: TAPJFNN {response.model_info.tapjfnn_loaded ? "✓" : "(not loaded, using MiniLM-only fallback)"} ·
        GNN {response.model_info.gnn_loaded ? "✓" : "(not trained, using TAPJFNN / cosine)"}
      </div>
      {response.ga && (
        <div className="mb-4 bg-slate-900 rounded-lg p-3 text-sm">
          <div className="font-semibold text-slate-200 mb-1">
            GA selected {response.ga.selected_ids.length} candidate(s) · fitness {response.ga.fitness.toFixed(3)}
          </div>
          {response.ga.selected_ids.length === 0 && (
            <div className="text-amber-300">No feasible set under current constraints. Loosen budget / diversity / min-fit.</div>
          )}
        </div>
      )}
      <ol className="space-y-3">
        {response.ranked.map((c) => {
          const selectedByGA = selected.has(c.id);
          return (
            <li
              key={c.id}
              className={
                "rounded-xl border p-4 " +
                (selectedByGA
                  ? "border-emerald-500 bg-emerald-900/30"
                  : "border-slate-700 bg-slate-900")
              }
            >
              <div className="flex items-baseline justify-between">
                <div className="flex items-baseline gap-3">
                  <span className="text-xl font-bold">#{c.rank}</span>
                  <span className="text-slate-100 truncate max-w-md">{c.id}</span>
                  {selectedByGA && (
                    <span className="text-xs font-semibold text-emerald-300">
                      GA-selected
                    </span>
                  )}
                </div>
                <span className="font-mono text-lg">
                  {(c.fit_score * 100).toFixed(1)}%
                </span>
              </div>
              <EntityRow label="Matching" map={c.matching_entities} tone="good" />
              <EntityRow label="CV entities" map={c.resume_entities} tone="muted" />
              <EntityRow label="JD entities" map={c.job_entities} tone="muted" />
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function EntityRow({
  label,
  map,
  tone,
}: {
  label: string;
  map: Record<string, string[]>;
  tone: "good" | "muted";
}) {
  const entries = Object.entries(map).filter(([, v]) => v && v.length);
  if (entries.length === 0) return null;
  const pillColor =
    tone === "good"
      ? "bg-emerald-800/60 text-emerald-200"
      : "bg-slate-800 text-slate-300";
  return (
    <div className="mt-3 text-sm">
      <div className="text-xs uppercase tracking-wider text-slate-400 mb-1">
        {label}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {entries.map(([cat, vals]) =>
          vals.slice(0, 8).map((v, i) => (
            <span
              key={`${cat}-${i}-${v}`}
              className={`px-2 py-0.5 rounded-full ${pillColor}`}
              title={cat}
            >
              {v}
            </span>
          ))
        )}
      </div>
    </div>
  );
}
