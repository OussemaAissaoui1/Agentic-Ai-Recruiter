import { createFileRoute } from "@tanstack/react-router";
import { Loader2, RefreshCw, RotateCcw, Sparkles } from "lucide-react";
import { toast } from "sonner";

import {
  useRecruiterProfile,
  useRefitProfile,
  useResetProfile,
} from "@/lib/queries";

export const Route = createFileRoute("/app/settings")({
  head: () => ({ meta: [{ title: "Settings — HireFlow" }] }),
  component: SettingsPage,
});

function SettingsPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <header>
        <h1 className="font-display text-3xl tracking-tight">Settings</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Configure how HireFlow adapts to you.
        </p>
      </header>

      <YourTasteSection />
    </div>
  );
}

function YourTasteSection() {
  const { data: profile, isLoading, isError } = useRecruiterProfile();
  const refit = useRefitProfile();
  const reset = useResetProfile();

  const onRefit = async () => {
    try {
      await refit.mutateAsync();
      toast.success("Your taste profile was retrained.");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  };

  const onReset = async () => {
    if (!confirm("Reset your taste profile? Your decision history is kept.")) {
      return;
    }
    try {
      await reset.mutateAsync();
      toast.success("Profile reset. The model will rebuild as you decide.");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <section className="rounded-2xl border border-border bg-background p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-accent" />
            <h2 className="text-lg font-semibold">Your taste</h2>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            HireFlow learns what you weight from your approve / reject
            decisions and surfaces a "recruiter fit" pill alongside the AI
            recommendation. Your decision history is private to your
            account.
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button
            onClick={onRefit}
            disabled={refit.isPending}
            className="inline-flex items-center gap-1.5 rounded-full border border-border px-3 py-1.5 text-xs font-medium hover:bg-muted disabled:opacity-50"
            title="Retrain now"
          >
            {refit.isPending ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
            Refit now
          </button>
          <button
            onClick={onReset}
            disabled={reset.isPending}
            className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium text-muted-foreground hover:text-destructive disabled:opacity-50"
            title="Reset profile"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset
          </button>
        </div>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        <Stat
          label="Decisions captured"
          value={profile ? String(profile.n_decisions) : "—"}
        />
        <Stat
          label="Confidence"
          value={profile ? confidenceLabel(profile.confidence) : "—"}
          accent={profile?.confidence === "warm"}
        />
        <Stat
          label="Profile version"
          value={profile && profile.version > 0 ? `v${profile.version}` : "—"}
        />
      </div>

      <div className="mt-5">
        {isLoading ? (
          <div className="text-sm text-muted-foreground">
            Loading your profile…
          </div>
        ) : isError ? (
          <div className="text-sm text-destructive">
            Couldn't load your profile.
          </div>
        ) : !profile || profile.n_decisions === 0 ? (
          <p className="text-sm italic text-muted-foreground">
            No decisions captured yet. Approve or reject a few candidates and
            the model will start to learn what you weight.
          </p>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2">
            <FactorList
              heading="Pulled toward approve"
              factors={profile.top_positive}
              emptyText="(no clear positive factors yet)"
              positive
            />
            <FactorList
              heading="Pulled toward reject"
              factors={profile.top_negative}
              emptyText="(no clear negative factors yet)"
              positive={false}
            />
          </div>
        )}
      </div>
    </section>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="rounded-xl border border-border bg-muted/30 p-3">
      <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <div
        className={`mt-1 font-mono text-lg font-semibold ${accent ? "text-accent" : ""}`}
      >
        {value}
      </div>
    </div>
  );
}

function FactorList({
  heading,
  factors,
  emptyText,
  positive,
}: {
  heading: string;
  factors: { label: string; weight: number }[];
  emptyText: string;
  positive: boolean;
}) {
  return (
    <div className="rounded-xl border border-border p-4">
      <div className="text-[10px] uppercase tracking-widest text-muted-foreground">
        {heading}
      </div>
      {factors.length === 0 ? (
        <div className="mt-3 text-xs italic text-muted-foreground">
          {emptyText}
        </div>
      ) : (
        <ul className="mt-3 space-y-2 text-sm">
          {factors.map((f) => (
            <li key={f.label}>
              <div className="flex items-baseline justify-between gap-2">
                <span>{f.label}</span>
                <span className="font-mono text-xs text-muted-foreground">
                  {positive ? "+" : "−"}
                  {Math.abs(f.weight).toFixed(2)}
                </span>
              </div>
              <div className="mt-1 h-1 overflow-hidden rounded-full bg-muted">
                <div
                  className={`h-full ${positive ? "bg-accent" : "bg-destructive"}`}
                  style={{
                    width: `${Math.min(100, Math.abs(f.weight) * 30)}%`,
                  }}
                />
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function confidenceLabel(confidence: "cold" | "warming" | "warm"): string {
  if (confidence === "cold") return "Learning";
  if (confidence === "warming") return "Warming up";
  return "Warm";
}
