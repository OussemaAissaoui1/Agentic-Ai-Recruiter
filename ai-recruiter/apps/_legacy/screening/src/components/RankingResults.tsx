import { useState } from "react";
import { RankResponse, parseCv } from "../api";

export function RankingResults({
  response,
  files,
  jobText,
}: {
  response: RankResponse;
  files: File[];
  jobText: string;
}) {
  const selected = new Set(response.ga?.selected_ids ?? []);
  const [interviewing, setInterviewing] = useState<string | null>(null);
  const [interviewError, setInterviewError] = useState<string | null>(null);

  async function startInterview(candidateId: string) {
    setInterviewError(null);
    const file = files.find((f) => f.name === candidateId);
    if (!file) {
      setInterviewError(`Original file not found in browser memory: ${candidateId}`);
      return;
    }
    setInterviewing(candidateId);
    try {
      const { text } = await parseCv(file);
      const candidateName =
        candidateId.replace(/\.[^.]+$/, "").replace(/[_\-]+/g, " ").trim() ||
        "Candidate";
      // Pull a job role guess from the JD's first line, fallback to default.
      const firstLine = jobText.split("\n").map((s) => s.trim()).find(Boolean) || "";
      const role =
        firstLine.length > 0 && firstLine.length < 80
          ? firstLine
          : "AI Engineering Intern";

      sessionStorage.setItem(
        "ai-recruiter:interview-payload",
        JSON.stringify({
          cv: text,
          jobDescription: jobText,
          role,
          candidateName,
          fitScore: response.ranked.find((c) => c.id === candidateId)?.fit_score ?? null,
          ts: Date.now(),
        })
      );
      window.location.href = "/interview/";
    } catch (e: any) {
      setInterviewError(e?.message || String(e));
      setInterviewing(null);
    }
  }

  return (
    <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg font-semibold text-slate-900">Ranking</h2>
        <div className="text-xs text-slate-500">
          model: TAPJFNN {response.model_info.tapjfnn_loaded ? "✓" : "(fallback)"} ·
          GNN {response.model_info.gnn_loaded ? "✓" : "(fallback)"}
        </div>
      </div>

      {response.ga && (
        <div className="mb-4 mt-3 bg-emerald-50 border border-emerald-200 rounded-xl p-3 text-sm">
          <div className="font-semibold text-emerald-900 mb-1">
            GA selected {response.ga.selected_ids.length} candidate(s) · fitness{" "}
            {response.ga.fitness.toFixed(3)}
          </div>
          {response.ga.selected_ids.length === 0 && (
            <div className="text-amber-700">
              No feasible set under current constraints. Loosen budget / diversity / min-fit.
            </div>
          )}
        </div>
      )}

      {interviewError && (
        <div className="mb-4 bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg p-3">
          {interviewError}
        </div>
      )}

      <ol className="space-y-3 mt-4">
        {response.ranked.map((c) => {
          const selectedByGA = selected.has(c.id);
          const isInterviewing = interviewing === c.id;
          return (
            <li
              key={c.id}
              className={
                "rounded-xl border p-4 transition-all " +
                (selectedByGA
                  ? "border-emerald-400 bg-emerald-50"
                  : "border-slate-200 bg-slate-50 hover:bg-white hover:shadow-sm")
              }
            >
              <div className="flex items-baseline justify-between gap-3 flex-wrap">
                <div className="flex items-baseline gap-3 min-w-0">
                  <span className="text-xl font-bold text-slate-900">#{c.rank}</span>
                  <span className="text-slate-800 truncate">{c.id}</span>
                  {selectedByGA && (
                    <span className="text-xs font-semibold text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded-full">
                      GA-selected
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <span className="font-mono text-lg text-slate-900">
                    {(c.fit_score * 100).toFixed(1)}%
                  </span>
                  <button
                    onClick={() => startInterview(c.id)}
                    disabled={!!interviewing}
                    className={
                      "text-sm font-semibold px-3.5 py-1.5 rounded-lg transition-all " +
                      (isInterviewing
                        ? "bg-indigo-100 text-indigo-700 cursor-wait"
                        : interviewing
                          ? "bg-slate-100 text-slate-400 cursor-not-allowed"
                          : "bg-indigo-600 text-white hover:bg-indigo-500 shadow-sm hover:shadow")
                    }
                    title="Start an interview with this candidate"
                  >
                    {isInterviewing ? "Preparing…" : "Interview →"}
                  </button>
                </div>
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
      ? "bg-emerald-100 text-emerald-800 border border-emerald-200"
      : "bg-white text-slate-600 border border-slate-200";
  return (
    <div className="mt-3 text-sm">
      <div className="text-xs uppercase tracking-wider text-slate-400 mb-1.5">
        {label}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {entries.map(([cat, vals]) =>
          vals.slice(0, 8).map((v, i) => (
            <span
              key={`${cat}-${i}-${v}`}
              className={`px-2 py-0.5 rounded-full text-xs ${pillColor}`}
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
