// TanStack Query hooks wrapping the typed api client.

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationOptions,
} from "@tanstack/react-query";
import {
  applications,
  health,
  interviews,
  jobs,
  notifications,
  scoring,
  type Application,
  type ApplicationStage,
  type Interview,
  type InterviewReport,
  type InterviewTurn,
  type Job,
  type Notification,
  type StatusResponse,
} from "@/lib/api";

// ─── Keys ────────────────────────────────────────────────────────────────────
export const qk = {
  status: ["status"] as const,
  jobs: (params?: { status?: "open" | "closed" }) =>
    ["jobs", params ?? {}] as const,
  job: (id: string) => ["jobs", id] as const,
  applications: (params?: {
    job_id?: string;
    stage?: string;
    candidate_email?: string;
  }) => ["applications", params ?? {}] as const,
  application: (id: string) => ["applications", id] as const,
  notifications: (params?: {
    user_role?: string;
    user_id?: string;
    unread_only?: boolean;
  }) => ["notifications", params ?? {}] as const,
  interviewsByApp: (application_id: string) =>
    ["interviews", { application_id }] as const,
  interview: (interview_id: string) =>
    ["interview", interview_id] as const,
  scoringReport: (interview_id: string) =>
    ["scoring", interview_id, "report"] as const,
};

// ─── Status / Health ─────────────────────────────────────────────────────────
export function useStatus() {
  return useQuery<StatusResponse>({
    queryKey: qk.status,
    queryFn: ({ signal }) => health.status({ signal }),
    refetchInterval: 30_000,
    retry: 0,
  });
}

// ─── Jobs ────────────────────────────────────────────────────────────────────
export function useJobs(params: { status?: "open" | "closed" } = {}) {
  return useQuery<Job[]>({
    queryKey: qk.jobs(params),
    queryFn: ({ signal }) => jobs.list(params, { signal }),
  });
}

export function useJob(id: string | undefined) {
  return useQuery<Job>({
    queryKey: qk.job(id ?? ""),
    queryFn: ({ signal }) => jobs.get(id as string, { signal }),
    enabled: !!id,
  });
}

export function useCreateJob(
  options?: UseMutationOptions<Job, Error, Partial<Job>>,
) {
  const qc = useQueryClient();
  return useMutation<Job, Error, Partial<Job>>({
    mutationFn: (body) => jobs.create(body),
    ...options,
    onSuccess: (data, vars, ctx) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      // Forward to the caller-provided handler if any.
      // Cast to bypass the variadic signature differences.
      const cb = options?.onSuccess as
        | ((data: Job, vars: Partial<Job>, ctx: unknown) => unknown)
        | undefined;
      cb?.(data, vars, ctx);
    },
  });
}

export function useUpdateJob(id: string) {
  const qc = useQueryClient();
  return useMutation<Job, Error, Partial<Job>>({
    mutationFn: (body) => jobs.update(id, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

export function useDeleteJob() {
  const qc = useQueryClient();
  return useMutation<void, Error, string>({
    mutationFn: (id) => jobs.remove(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

// ─── Applications ────────────────────────────────────────────────────────────
export function useApplications(
  params: { job_id?: string; stage?: string; candidate_email?: string } = {},
) {
  return useQuery<Application[]>({
    queryKey: qk.applications(params),
    queryFn: ({ signal }) => applications.list(params, { signal }),
  });
}

export function useApplication(id: string | undefined) {
  return useQuery<Application>({
    queryKey: qk.application(id ?? ""),
    queryFn: ({ signal }) => applications.get(id as string, { signal }),
    enabled: !!id,
    refetchOnWindowFocus: true,
    // Poll every 15s only while the candidate is waiting for review.
    // Once approved/rejected/interviewed/etc the result is terminal for the
    // gate and we stop polling to keep load near zero.
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.stage === "applied" ? 15_000 : false;
    },
  });
}

export type CreateApplicationInput = {
  job_id: string;
  candidate_name?: string;
  candidate_email?: string;
  cv: File;
};

export function useCreateApplication(
  options?: UseMutationOptions<Application, Error, CreateApplicationInput>,
) {
  const qc = useQueryClient();
  return useMutation<Application, Error, CreateApplicationInput>({
    mutationFn: (body) => applications.create(body),
    ...options,
    onSuccess: (data, vars, ctx) => {
      qc.invalidateQueries({ queryKey: ["applications"] });
      const cb = options?.onSuccess as
        | ((data: Application, vars: CreateApplicationInput, ctx: unknown) => unknown)
        | undefined;
      cb?.(data, vars, ctx);
    },
  });
}

export function useUpdateApplication() {
  const qc = useQueryClient();
  return useMutation<
    Application,
    Error,
    { id: string; body: { stage?: ApplicationStage; notes?: string } }
  >({
    mutationFn: ({ id, body }) => applications.update(id, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["applications"] });
    },
  });
}

export function useInviteInterview() {
  const qc = useQueryClient();
  return useMutation<{ ok: boolean } | Application, Error, string>({
    mutationFn: (id) => applications.inviteInterview(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["applications"] });
    },
  });
}

// ─── Notifications ───────────────────────────────────────────────────────────
export function useNotifications(
  params: { user_role?: string; user_id?: string; unread_only?: boolean } = {},
) {
  return useQuery<Notification[]>({
    queryKey: qk.notifications(params),
    queryFn: ({ signal }) => notifications.list(params, { signal }),
    refetchInterval: 60_000,
  });
}

export function useMarkNotificationRead() {
  const qc = useQueryClient();
  return useMutation<Notification, Error, string>({
    mutationFn: (id) => notifications.markRead(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["notifications"] });
    },
  });
}

// ─── Interviews ──────────────────────────────────────────────────────────────
export function useInterviewsByApplication(application_id: string | undefined) {
  return useQuery<Interview[]>({
    queryKey: qk.interviewsByApp(application_id ?? ""),
    queryFn: ({ signal }) =>
      interviews.list({ application_id }, { signal }),
    enabled: !!application_id,
  });
}

export function useInterview(interview_id: string | undefined) {
  return useQuery<Interview>({
    queryKey: qk.interview(interview_id ?? ""),
    queryFn: ({ signal }) =>
      interviews.get(interview_id as string, { signal }),
    enabled: !!interview_id,
  });
}

export function useCreateInterview() {
  const qc = useQueryClient();
  return useMutation<
    Interview,
    Error,
    {
      application_id: string;
      transcript: InterviewTurn[];
      status?: string;
      started_at?: number;
      ended_at?: number;
    }
  >({
    mutationFn: (body) => interviews.create(body),
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({
        queryKey: qk.interviewsByApp(vars.application_id),
      });
      qc.invalidateQueries({ queryKey: ["applications"] });
    },
  });
}

// ─── Scoring ─────────────────────────────────────────────────────────────────
export function useScoringReport(interview_id: string | undefined) {
  return useQuery<InterviewReport | null>({
    queryKey: qk.scoringReport(interview_id ?? ""),
    queryFn: async ({ signal }) => {
      try {
        return await scoring.report(interview_id as string, { signal });
      } catch (err) {
        // 404 = report not yet generated; surface as null instead of throwing
        const status = (err as { status?: number } | null)?.status;
        if (status === 404) return null;
        throw err;
      }
    },
    enabled: !!interview_id,
    retry: (failureCount, err) => {
      const status = (err as { status?: number } | null)?.status;
      if (status === 404) return false;
      return failureCount < 2;
    },
  });
}

export function useRunScoring() {
  const qc = useQueryClient();
  return useMutation<
    InterviewReport,
    Error,
    { interview_id: string; force?: boolean; transcript?: InterviewTurn[] }
  >({
    mutationFn: ({ interview_id, force, transcript }) =>
      scoring.run(interview_id, { force, transcript }),
    onSuccess: (data) => {
      qc.setQueryData(qk.scoringReport(data.interview_id), data);
      qc.invalidateQueries({ queryKey: ["interviews"] });
      qc.invalidateQueries({ queryKey: ["applications"] });
    },
  });
}

export function useDeleteScoringReport() {
  const qc = useQueryClient();
  return useMutation<{ deleted: boolean }, Error, string>({
    mutationFn: (interview_id) => scoring.deleteReport(interview_id),
    onSuccess: (_data, interview_id) => {
      qc.setQueryData(qk.scoringReport(interview_id), null);
    },
  });
}
