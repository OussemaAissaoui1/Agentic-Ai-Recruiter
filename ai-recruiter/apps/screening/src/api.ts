export interface RankedCandidate {
  rank: number;
  id: string;
  fit_score: number;
  matching_entities: Record<string, string[]>;
  resume_entities: Record<string, string[]>;
  job_entities: Record<string, string[]>;
}

export interface GAResult {
  selected_ids: string[];
  fitness: number;
  history: number[];
}

export interface RankResponse {
  ranked: RankedCandidate[];
  model_info: {
    tapjfnn_loaded: boolean;
    gnn_loaded: boolean;
  };
  ga?: GAResult;
}

export interface CandidateMetadata {
  id: string;
  cost: number;
  gender_female: 0 | 1;
  interview_score: number;
  cultural_fit: number;
  experience_score: number;
  salary_alignment: number;
}

export interface GAConstraints {
  budget: number;
  min_female_ratio: number;
  min_fit_threshold: number;
  role_requirements: Record<string, number>;
}

export async function rankCandidates(
  jobDescription: string,
  files: File[],
  applyGA: boolean,
  metadata: CandidateMetadata[],
  constraints: GAConstraints
): Promise<RankResponse> {
  const form = new FormData();
  form.append("job_description", jobDescription);
  form.append("apply_ga_flag", applyGA ? "true" : "false");
  form.append("metadata_json", JSON.stringify(metadata));
  form.append("constraints_json", JSON.stringify(constraints));
  for (const f of files) form.append("files", f);
  const res = await fetch("/api/matching/rank", { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`rank failed: ${res.status} ${detail}`);
  }
  return res.json();
}

export async function healthCheck() {
  const res = await fetch("/api/matching/health");
  return res.json();
}

export interface ExtractedSignals {
  years_experience: number;
  experience_score: number;
  cultural_fit: number;
  salary_expectation_usd: number;
  salary_alignment: number;
  gender_female: 0 | 1;
  confidence: number;
  notes: string;
  error?: string | null;
}

export interface ExtractedEntry {
  metadata: CandidateMetadata;
  signals: ExtractedSignals;
  error: string | null;
}

export interface ExtractResponse {
  signals: Record<string, ExtractedEntry>;
  model: string;
}

export async function extractSignals(
  jobDescription: string,
  files: File[]
): Promise<ExtractResponse> {
  const form = new FormData();
  form.append("job_description", jobDescription);
  for (const f of files) form.append("files", f);
  const res = await fetch("/api/matching/extract-signals", { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`extract-signals failed: ${res.status} ${detail}`);
  }
  return res.json();
}
