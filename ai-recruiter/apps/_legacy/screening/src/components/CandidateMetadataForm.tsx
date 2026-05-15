import { Fragment } from "react";
import { CandidateMetadata, ExtractedSignals } from "../api";

export function CandidateMetadataForm({
  files,
  metadata,
  onChange,
  onAutofill,
  autofilling,
  canAutofill,
  extracted,
  autofillError,
  llmModel,
  llmLoaded,
  llmHasToken,
}: {
  files: File[];
  metadata: Record<string, CandidateMetadata>;
  onChange: (m: Record<string, CandidateMetadata>) => void;
  onAutofill: () => void | Promise<void>;
  autofilling: boolean;
  canAutofill: boolean;
  extracted: Record<string, ExtractedSignals>;
  autofillError: string | null;
  llmModel?: string;
  llmLoaded?: boolean;
  llmHasToken?: boolean;
}) {
  function update(id: string, patch: Partial<CandidateMetadata>) {
    const prev = metadata[id] || defaultsFor(id);
    onChange({ ...metadata, [id]: { ...prev, ...patch } });
  }

  if (files.length === 0) {
    return (
      <div className="bg-slate-800 rounded-xl p-5 shadow text-slate-400">
        Upload CVs first to fill in per-candidate metadata for the genetic algorithm.
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-xl p-5 shadow">
      <div className="flex items-start justify-between mb-3 gap-4">
        <div>
          <h2 className="text-lg font-semibold">Per-candidate metadata</h2>
          <p className="text-sm text-slate-400 mt-1">
            Interview score is filled by the hiring team. Everything else can be
            autofilled by the local LLM agent and edited inline.
          </p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <button
            onClick={() => onAutofill()}
            disabled={!canAutofill || autofilling}
            className="px-3 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-sm font-medium"
            title={
              llmHasToken === false
                ? "HF_TOKEN is not set on the backend. The extractor will fail to load."
                : canAutofill
                ? "Extract signals with the local LLM (first call loads the model, ~15 s)"
                : "Upload CVs and enter a job description first"
            }
          >
            {autofilling ? "Extracting…" : "Autofill with AI"}
          </button>
          {llmModel && (
            <span className="text-[11px] text-slate-500">
              model: {llmModel}{" "}
              {llmLoaded === false && llmHasToken === false ? (
                <span className="text-amber-400">(HF_TOKEN missing)</span>
              ) : llmLoaded === false ? (
                <span className="text-slate-500">(loads on first use)</span>
              ) : null}
            </span>
          )}
        </div>
      </div>

      {autofillError && (
        <div className="text-red-400 text-sm mb-3">{autofillError}</div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-slate-300">
            <tr className="border-b border-slate-700">
              <th className="text-left py-2 pr-3">CV filename</th>
              <th className="text-left py-2 pr-3">Cost (USD)</th>
              <th className="text-left py-2 pr-3">Gender</th>
              <th className="text-left py-2 pr-3">Interview</th>
              <th className="text-left py-2 pr-3">Cultural fit</th>
              <th className="text-left py-2 pr-3">Experience</th>
              <th className="text-left py-2 pr-3">Salary align.</th>
            </tr>
          </thead>
          <tbody>
            {files.map((f) => {
              const id = f.name;
              const m = metadata[id] || defaultsFor(id);
              const sig = extracted[id];
              return (
                <Fragment key={id}>
                  <tr className="border-b border-slate-800">
                    <td className="py-2 pr-3 text-slate-200 max-w-xs truncate">
                      <div>{id}</div>
                      {sig && (
                        <div className="text-[11px] text-emerald-400 mt-0.5">
                          AI-filled · conf {sig.confidence.toFixed(2)}
                        </div>
                      )}
                    </td>
                    <td className="py-2 pr-3">
                      <input
                        type="number"
                        className="w-28 bg-slate-900 border border-slate-700 rounded px-2 py-1"
                        min={0}
                        value={m.cost}
                        onChange={(e) => update(id, { cost: Number(e.target.value) })}
                      />
                    </td>
                    <td className="py-2 pr-3">
                      <select
                        className="bg-slate-900 border border-slate-700 rounded px-2 py-1"
                        value={m.gender_female}
                        onChange={(e) =>
                          update(id, { gender_female: Number(e.target.value) as 0 | 1 })
                        }
                      >
                        <option value={0}>Male / Other</option>
                        <option value={1}>Female</option>
                      </select>
                    </td>
                    {(["interview_score", "cultural_fit", "experience_score", "salary_alignment"] as const).map(
                      (k) => (
                        <td key={k} className="py-2 pr-3">
                          <input
                            type="number"
                            step={0.05}
                            min={0}
                            max={1}
                            className="w-20 bg-slate-900 border border-slate-700 rounded px-2 py-1"
                            value={m[k]}
                            onChange={(e) =>
                              update(id, { [k]: Number(e.target.value) } as Partial<CandidateMetadata>)
                            }
                          />
                        </td>
                      )
                    )}
                  </tr>
                  {sig && sig.notes && (
                    <tr className="border-b border-slate-800">
                      <td colSpan={7} className="py-1.5 pr-3 pl-3 text-[11px] text-slate-400 italic">
                        {sig.notes}
                      </td>
                    </tr>
                  )}
                  {sig && sig.error && (
                    <tr className="border-b border-slate-800">
                      <td colSpan={7} className="py-1.5 pr-3 pl-3 text-[11px] text-red-400">
                        LLM error: {sig.error}
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function defaultsFor(id: string): CandidateMetadata {
  return {
    id,
    cost: 0,
    gender_female: 0,
    interview_score: 0,
    cultural_fit: 0,
    experience_score: 0,
    salary_alignment: 0,
  };
}
