// HireFlow API client. Plain fetch — vite proxies /api, /ws, /avatar to backend.
// All helpers support AbortSignal and throw ApiError on non-2xx.

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

export type Job = {
  id: string;
  title: string;
  team: string;
  location: string;
  work_mode: string;
  level: string;
  employment_type: string;
  salary_min: number;
  salary_max: number;
  currency: string;
  description: string;
  must_have: string[];
  nice_to_have: string[];
  status: "open" | "closed";
  created_at: string;
  updated_at: string;
};

export type ApplicationStage =
  | "applied"
  | "approved"
  | "interviewed"
  | "offer"
  | "hired"
  | "rejected";

export type Application = {
  id: string;
  job_id: string;
  candidate_name: string;
  candidate_email: string;
  cv_filename: string;
  cv_text?: string | null;
  fit_score: number; // 0-1
  matched_skills: string[];
  missing_skills: string[];
  stage: ApplicationStage;
  notes?: string | null;
  created_at: string;
};

export type Notification = {
  id: string;
  user_role: string;
  user_id: string;
  kind: string;
  title: string;
  body: string;
  link?: string | null;
  read: boolean;
  created_at: string;
};

export type StatusResponse = {
  ready: boolean;
  nlp_ready: boolean;
  avatar_ready: boolean;
  stt_ready: boolean;
  vad_ready: boolean;
};

export type GeneratedJD = {
  description: string;
  must_have: string[];
  nice_to_have: string[];
};

export type RankItem = {
  id: string;
  score: number;
  [k: string]: unknown;
};

export type RankResponse = {
  ranked: RankItem[];
  model_info?: Record<string, unknown>;
  ga?: Record<string, unknown>;
};

export type ParseResponse = { id: string; text: string };

export type NlpSessionResponse = { session_id: string };

export type NlpChatResponse = {
  question: string;
  audio_b64?: string | null;
  [k: string]: unknown;
};

export type TtsResponse = { audio_b64: string };

export type TranscribeResponse = { text: string };

export type VisionSessionResponse = { session_id: string };

type FetchOpts = { signal?: AbortSignal };

async function request<T>(
  url: string,
  init: RequestInit & FetchOpts = {},
): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: {
      ...(init.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
      ...(init.headers || {}),
    },
    signal: init.signal,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const data = await res.json();
      if (data && typeof data === "object") {
        msg = (data as { detail?: string; message?: string }).detail
          ?? (data as { message?: string }).message
          ?? JSON.stringify(data);
      }
    } catch {
      try {
        msg = await res.text();
      } catch {
        /* noop */
      }
    }
    throw new ApiError(res.status, msg || `HTTP ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return (await res.json()) as T;
  }
  return (await res.text()) as unknown as T;
}

function qs(params: Record<string, string | number | boolean | undefined | null>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    sp.append(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

// ─── Health ──────────────────────────────────────────────────────────────────
export const health = {
  status: (opts?: FetchOpts) => request<StatusResponse>("/api/status", { signal: opts?.signal }),
  matching: (opts?: FetchOpts) =>
    request<Record<string, unknown>>("/api/matching/health", { signal: opts?.signal }),
};

// ─── Jobs ────────────────────────────────────────────────────────────────────
export const jobs = {
  list: (
    params: { status?: "open" | "closed" } = {},
    opts?: FetchOpts,
  ) => request<Job[]>(`/api/recruit/jobs${qs(params)}`, { signal: opts?.signal }),

  get: (id: string, opts?: FetchOpts) =>
    request<Job>(`/api/recruit/jobs/${encodeURIComponent(id)}`, { signal: opts?.signal }),

  create: (body: Partial<Job>, opts?: FetchOpts) =>
    request<Job>("/api/recruit/jobs", {
      method: "POST",
      body: JSON.stringify(body),
      signal: opts?.signal,
    }),

  update: (id: string, body: Partial<Job>, opts?: FetchOpts) =>
    request<Job>(`/api/recruit/jobs/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
      signal: opts?.signal,
    }),

  remove: (id: string, opts?: FetchOpts) =>
    request<void>(`/api/recruit/jobs/${encodeURIComponent(id)}`, {
      method: "DELETE",
      signal: opts?.signal,
    }),

  /** SSE generator. Yields parsed JSON events.
   *  Final event payload is `{jd: {description, must_have, nice_to_have}}`.
   *  Token events look like `{token: "..."}`. */
  generateStream: async function* (
    body: { title: string; team?: string; seed?: string; level?: string; location?: string },
    opts?: FetchOpts,
  ): AsyncGenerator<{ token?: string; jd?: GeneratedJD; [k: string]: unknown }, void, void> {
    const res = await fetch("/api/recruit/jobs/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal: opts?.signal,
    });
    if (!res.ok || !res.body) {
      throw new ApiError(res.status, `JD stream failed: ${res.statusText}`);
    }
    yield* parseSSE(res.body, opts?.signal);
  },
};

// ─── Applications ────────────────────────────────────────────────────────────
export const applications = {
  list: (
    params: { job_id?: string; stage?: string; candidate_email?: string } = {},
    opts?: FetchOpts,
  ) =>
    request<Application[]>(`/api/recruit/applications${qs(params)}`, {
      signal: opts?.signal,
    }),

  get: (id: string, opts?: FetchOpts) =>
    request<Application>(`/api/recruit/applications/${encodeURIComponent(id)}`, {
      signal: opts?.signal,
    }),

  create: (
    body: {
      job_id: string;
      candidate_name?: string;
      candidate_email?: string;
      cv: File;
    },
    opts?: FetchOpts,
  ) => {
    const fd = new FormData();
    fd.append("job_id", body.job_id);
    if (body.candidate_name) fd.append("candidate_name", body.candidate_name);
    if (body.candidate_email) fd.append("candidate_email", body.candidate_email);
    fd.append("cv", body.cv);
    return request<Application>("/api/recruit/applications", {
      method: "POST",
      body: fd,
      signal: opts?.signal,
    });
  },

  update: (
    id: string,
    body: { stage?: ApplicationStage; notes?: string },
    opts?: FetchOpts,
  ) =>
    request<Application>(`/api/recruit/applications/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify(body),
      signal: opts?.signal,
    }),

  inviteInterview: (id: string, opts?: FetchOpts) =>
    request<{ ok: boolean } | Application>(
      `/api/recruit/applications/${encodeURIComponent(id)}/invite-interview`,
      { method: "POST", signal: opts?.signal },
    ),
};

// ─── Notifications ───────────────────────────────────────────────────────────
export const notifications = {
  list: (
    params: { user_role?: string; user_id?: string; unread_only?: boolean } = {},
    opts?: FetchOpts,
  ) =>
    request<Notification[]>(`/api/recruit/notifications${qs(params)}`, {
      signal: opts?.signal,
    }),

  markRead: (id: string, opts?: FetchOpts) =>
    request<Notification>(`/api/recruit/notifications/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ read: true }),
      signal: opts?.signal,
    }),
};

// ─── Matching ────────────────────────────────────────────────────────────────
export const matching = {
  parse: (file: File, opts?: FetchOpts) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<ParseResponse>("/api/matching/parse", {
      method: "POST",
      body: fd,
      signal: opts?.signal,
    });
  },

  rank: (
    body: {
      job_description: string;
      files: File[];
      apply_ga_flag?: boolean;
      metadata_json?: string;
      constraints_json?: string;
    },
    opts?: FetchOpts,
  ) => {
    const fd = new FormData();
    fd.append("job_description", body.job_description);
    body.files.forEach((f) => fd.append("files", f));
    if (body.apply_ga_flag !== undefined)
      fd.append("apply_ga_flag", String(body.apply_ga_flag));
    if (body.metadata_json) fd.append("metadata_json", body.metadata_json);
    if (body.constraints_json) fd.append("constraints_json", body.constraints_json);
    return request<RankResponse>("/api/matching/rank", {
      method: "POST",
      body: fd,
      signal: opts?.signal,
    });
  },
};

// ─── NLP ─────────────────────────────────────────────────────────────────────
export const nlp = {
  session: (body: { cv_text: string; job_role: string }, opts?: FetchOpts) =>
    request<NlpSessionResponse>("/api/nlp/session", {
      method: "POST",
      body: JSON.stringify(body),
      signal: opts?.signal,
    }),

  chat: (
    body: {
      session_id: string;
      cv_text: string;
      job_role: string;
      answer: string;
      history: Array<{ role: "user" | "assistant"; content: string }>;
    },
    opts?: FetchOpts,
  ) =>
    request<NlpChatResponse>("/api/nlp/chat", {
      method: "POST",
      body: JSON.stringify(body),
      signal: opts?.signal,
    }),

  /** SSE stream of question tokens. Pass any params the backend wants via `params`. */
  stream: async function* (
    params: Record<string, string>,
    opts?: FetchOpts,
  ): AsyncGenerator<{ token?: string; done?: boolean; [k: string]: unknown }, void, void> {
    const sp = new URLSearchParams(params).toString();
    const res = await fetch(`/api/nlp/stream${sp ? `?${sp}` : ""}`, {
      headers: { Accept: "text/event-stream" },
      signal: opts?.signal,
    });
    if (!res.ok || !res.body) {
      throw new ApiError(res.status, `NLP stream failed: ${res.statusText}`);
    }
    yield* parseSSE(res.body, opts?.signal);
  },

  tts: (text: string, opts?: FetchOpts) =>
    request<TtsResponse>("/api/nlp/tts", {
      method: "POST",
      body: JSON.stringify({ text }),
      signal: opts?.signal,
    }),
};

// ─── Vision ──────────────────────────────────────────────────────────────────
export const vision = {
  startSession: (body: { candidate_id?: string } = {}, opts?: FetchOpts) =>
    request<VisionSessionResponse>("/api/vision/session/start", {
      method: "POST",
      body: JSON.stringify(body),
      signal: opts?.signal,
    }),

  pushQuestion: (sessionId: string, text: string, opts?: FetchOpts) =>
    request<{ ok: boolean }>(
      `/api/vision/session/${encodeURIComponent(sessionId)}/question`,
      {
        method: "POST",
        body: JSON.stringify({ text }),
        signal: opts?.signal,
      },
    ),

  /** WebSocket URL helper for the vision frame socket. */
  wsUrl: (sessionId: string) => {
    const proto = typeof window !== "undefined" && window.location.protocol === "https:" ? "wss" : "ws";
    const host = typeof window !== "undefined" ? window.location.host : "localhost";
    return `${proto}://${host}/ws/vision?session_id=${encodeURIComponent(sessionId)}`;
  },
};

// ─── Transcribe ──────────────────────────────────────────────────────────────
export const transcribe = {
  http: (file: File | Blob, opts?: FetchOpts) => {
    const fd = new FormData();
    fd.append("file", file, "audio.webm");
    return request<TranscribeResponse>("/api/transcribe", {
      method: "POST",
      body: fd,
      signal: opts?.signal,
    });
  },

  wsUrl: () => {
    const proto = typeof window !== "undefined" && window.location.protocol === "https:" ? "wss" : "ws";
    const host = typeof window !== "undefined" ? window.location.host : "localhost";
    return `${proto}://${host}/ws/transcribe`;
  },
};

// ─── SSE parser ──────────────────────────────────────────────────────────────
async function* parseSSE<T = Record<string, unknown>>(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<T, void, void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    while (true) {
      if (signal?.aborted) break;
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // Split on double newline (SSE event boundary)
      let idx;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const raw = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const ev = parseSSEEvent(raw);
        if (ev !== null) yield ev as T;
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      /* noop */
    }
  }
}

function parseSSEEvent(raw: string): unknown | null {
  const lines = raw.split(/\r?\n/);
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (!dataLines.length) return null;
  const data = dataLines.join("\n");
  if (!data || data === "[DONE]") return null;
  try {
    return JSON.parse(data);
  } catch {
    // Plain text token
    return { token: data };
  }
}

export const api = {
  health,
  jobs,
  applications,
  notifications,
  matching,
  nlp,
  vision,
  transcribe,
};
