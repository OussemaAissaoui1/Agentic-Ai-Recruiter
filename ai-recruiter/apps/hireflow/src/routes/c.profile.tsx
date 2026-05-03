import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Upload, Save, Loader2, FileText } from "lucide-react";
import { useApp } from "@/lib/store";
import { matching } from "@/lib/api";
import { toast } from "sonner";

export const Route = createFileRoute("/c/profile")({
  head: () => ({ meta: [{ title: "Profile — HireFlow" }] }),
  component: Profile,
});

function Profile() {
  const { profile, setProfile, setCandidate } = useApp();
  const [local, setLocal] = useState(profile);
  const [uploading, setUploading] = useState(false);

  const onUpload = async (file: File) => {
    setUploading(true);
    try {
      const res = await matching.parse(file);
      const next = { cvFilename: file.name, cvText: res.text };
      setLocal((s) => ({ ...s, ...next }));
      setProfile(next);
      toast.success("CV parsed.");
    } catch (e) {
      toast.error(`CV parse failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setUploading(false);
    }
  };

  const save = () => {
    setProfile(local);
    setCandidate({ name: local.candidateName, email: local.candidateEmail });
    toast.success("Profile saved.");
  };

  // Crude profile strength heuristic: count filled fields out of 7.
  const filled = [
    local.candidateName,
    local.candidateEmail,
    local.about,
    local.skills,
    local.years,
    local.lookingFor,
    local.cvText,
  ].filter((v) => !!(v && String(v).trim())).length;
  const strength = Math.round((filled / 7) * 100);

  return (
    <div className="space-y-8">
      <div className="flex items-end justify-between">
        <h1 className="font-display text-4xl tracking-tight">Your profile</h1>
        <button
          onClick={save}
          className="inline-flex items-center gap-2 rounded-full bg-violet-grad px-4 py-2 text-sm font-semibold text-accent-foreground shadow-glow"
        >
          <Save className="h-4 w-4" /> Save
        </button>
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <div className="space-y-4 rounded-2xl border border-border bg-card p-6 md:col-span-2">
          <Field label="Name">
            <input
              value={local.candidateName}
              onChange={(e) => setLocal({ ...local, candidateName: e.target.value })}
              placeholder="Aria Park"
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
            />
          </Field>
          <Field label="Email">
            <input
              value={local.candidateEmail}
              onChange={(e) => setLocal({ ...local, candidateEmail: e.target.value })}
              type="email"
              placeholder="aria@example.com"
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
            />
          </Field>
          <Field label="About">
            <textarea
              value={local.about}
              onChange={(e) => setLocal({ ...local, about: e.target.value })}
              rows={4}
              className="w-full resize-none rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
            />
          </Field>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label="Skills">
              <input
                value={local.skills}
                onChange={(e) => setLocal({ ...local, skills: e.target.value })}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </Field>
            <Field label="Years">
              <input
                value={local.years}
                onChange={(e) => setLocal({ ...local, years: e.target.value })}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </Field>
            <Field label="Looking for">
              <input
                value={local.lookingFor}
                onChange={(e) => setLocal({ ...local, lookingFor: e.target.value })}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </Field>
            <Field label="Comp">
              <input
                value={local.comp}
                onChange={(e) => setLocal({ ...local, comp: e.target.value })}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent"
              />
            </Field>
          </div>
        </div>
        <div className="space-y-4">
          <div className="rounded-2xl border border-border bg-card p-6">
            <div className="text-xs uppercase tracking-widest text-muted-foreground">Profile strength</div>
            <div className="mt-2 font-display text-4xl">{strength}%</div>
            <div className="mt-1 text-xs text-muted-foreground">
              {strength >= 90 ? "Looking great." : "Add more details to lift your match scores."}
            </div>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
              <div className="h-full bg-violet-grad" style={{ width: `${strength}%` }} />
            </div>
          </div>
          <div className="rounded-2xl border border-border bg-card p-6">
            <div className="text-xs uppercase tracking-widest text-muted-foreground">CV</div>
            <label className="mt-3 flex cursor-pointer items-center justify-between rounded-xl border border-dashed border-border bg-muted/30 p-4 text-sm text-muted-foreground transition hover:bg-muted/50">
              <span className="inline-flex items-center gap-2">
                {uploading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Upload className="h-4 w-4" />
                )}
                {local.cvFilename || "Upload CV (PDF, DOCX, TXT, image)"}
              </span>
              <input
                type="file"
                accept=".pdf,.docx,.txt,.png,.jpg,.jpeg"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onUpload(f);
                }}
                className="hidden"
                disabled={uploading}
              />
            </label>
            {local.cvText && (
              <div className="mt-3 rounded-xl border border-border bg-background p-3 text-xs">
                <div className="mb-1 inline-flex items-center gap-1 text-muted-foreground">
                  <FileText className="h-3 w-3" /> Parsed preview
                </div>
                <pre className="max-h-40 overflow-y-auto whitespace-pre-wrap font-mono text-[11px] leading-relaxed">
                  {local.cvText.slice(0, 800)}
                  {local.cvText.length > 800 ? "…" : ""}
                </pre>
              </div>
            )}
            {/* TODO: persist profile (and parsed CV) to a real backend endpoint when one exists. */}
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs uppercase tracking-widest text-muted-foreground">{label}</span>
      <div className="mt-1.5">{children}</div>
    </label>
  );
}
