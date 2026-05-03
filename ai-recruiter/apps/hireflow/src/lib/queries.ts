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
  jobs,
  notifications,
  type Application,
  type ApplicationStage,
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
