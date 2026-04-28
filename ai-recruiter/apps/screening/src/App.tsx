import { useEffect, useMemo, useState } from "react";
import {
  CandidateMetadata,
  ExtractedSignals,
  GAConstraints,
  RankResponse,
  extractSignals,
  healthCheck,
  rankCandidates,
} from "./api";
import { JobDescriptionInput } from "./components/JobDescriptionInput";
import { CVUpload } from "./components/CVUpload";
import { CandidateMetadataForm } from "./components/CandidateMetadataForm";
import { GAConstraintsForm } from "./components/GAConstraintsForm";
import { RankingResults } from "./components/RankingResults";

export default function App() {
  const [jobText, setJobText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [metadata, setMetadata] = useState<Record<string, CandidateMetadata>>({});
  const [gaEnabled, setGaEnabled] = useState(false);
  const [constraints, setConstraints] = useState<GAConstraints>({
    budget: 0,
    min_female_ratio: 0,
    min_fit_threshold: 0,
    role_requirements: {},
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<RankResponse | null>(null);
  const [health, setHealth] = useState<any>(null);
  const [autofilling, setAutofilling] = useState(false);
  const [extracted, setExtracted] = useState<Record<string, ExtractedSignals>>({});
  const [extractError, setExtractError] = useState<string | null>(null);

  useEffect(() => {
    healthCheck().then(setHealth).catch(() => setHealth(null));
  }, []);

  // Prune metadata entries for removed files.
  useEffect(() => {
    const keep = new Set(files.map((f) => f.name));
    const next: Record<string, CandidateMetadata> = {};
    for (const [k, v] of Object.entries(metadata)) {
      if (keep.has(k)) next[k] = v;
    }
    if (Object.keys(next).length !== Object.keys(metadata).length) {
      setMetadata(next);
    }
  }, [files]);

  const canSubmit = useMemo(
    () => jobText.trim().length > 10 && files.length > 0 && !loading,
    [jobText, files, loading]
  );

  async function autofillFromLLM() {
    if (files.length === 0 || jobText.trim().length < 10) return;
    setAutofilling(true);
    setExtractError(null);
    try {
      const resp = await extractSignals(jobText, files);
      const nextMeta = { ...metadata };
      const nextExtracted: Record<string, ExtractedSignals> = { ...extracted };
      for (const [fname, entry] of Object.entries(resp.signals)) {
        nextMeta[fname] = { ...entry.metadata } as CandidateMetadata;
        nextExtracted[fname] = { ...entry.signals, error: entry.error };
      }
      setMetadata(nextMeta);
      setExtracted(nextExtracted);
    } catch (e: any) {
      setExtractError(e.message || String(e));
    } finally {
      setAutofilling(false);
    }
  }

  async function submit() {
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const metaArray: CandidateMetadata[] = files.map((f) =>
        metadata[f.name] || {
          id: f.name,
          cost: 0,
          gender_female: 0,
          interview_score: 0,
          cultural_fit: 0,
          experience_score: 0,
          salary_alignment: 0,
        }
      );
      const resp = await rankCandidates(
        jobText,
        files,
        gaEnabled,
        metaArray,
        constraints
      );
      setResponse(resp);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen">
      <header className="px-6 py-5 border-b border-slate-800 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">CV Ranking System</h1>
          <p className="text-sm text-slate-400">
            TAPJFNN (Qin 2020) + GNN (Frazzetto 2025) + CAGA (Malini 2026)
          </p>
        </div>
        <div className="text-xs text-slate-400">
          {health ? (
            <div>
              <span className={health.tapjfnn_loaded ? "text-emerald-400" : "text-slate-500"}>
                TAPJFNN {health.tapjfnn_loaded ? "loaded" : "not loaded"}
              </span>
              {" · "}
              <span className={health.gnn_loaded ? "text-emerald-400" : "text-slate-500"}>
                GNN {health.gnn_loaded ? "loaded" : "not loaded"}
              </span>
              {" · "}
              <span className={health.lda_loaded ? "text-emerald-400" : "text-slate-500"}>
                LDA {health.lda_loaded ? "loaded" : "not loaded"}
              </span>
            </div>
          ) : (
            <span>backend: offline</span>
          )}
        </div>
      </header>

      <main className="max-w-5xl mx-auto p-6 space-y-6">
        <JobDescriptionInput value={jobText} onChange={setJobText} />
        <CVUpload files={files} onChange={setFiles} />
        <GAConstraintsForm
          enabled={gaEnabled}
          setEnabled={setGaEnabled}
          constraints={constraints}
          onChange={setConstraints}
        />
        {gaEnabled && (
          <CandidateMetadataForm
            files={files}
            metadata={metadata}
            onChange={setMetadata}
            onAutofill={autofillFromLLM}
            autofilling={autofilling}
            canAutofill={files.length > 0 && jobText.trim().length >= 10}
            extracted={extracted}
            autofillError={extractError}
            llmModel={health?.llm?.model}
            llmLoaded={!!health?.llm?.loaded}
            llmHasToken={!!health?.llm?.has_token}
          />
        )}

        <div className="flex items-center justify-between">
          <button
            disabled={!canSubmit}
            onClick={submit}
            className="px-6 py-3 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed font-semibold"
          >
            {loading ? "Ranking…" : "Rank candidates"}
          </button>
          {error && <div className="text-red-400 text-sm ml-4">{error}</div>}
        </div>

        {response && <RankingResults response={response} />}
      </main>
    </div>
  );
}
