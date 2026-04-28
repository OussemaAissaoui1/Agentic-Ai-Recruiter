import { GAConstraints } from "../api";

export function GAConstraintsForm({
  enabled,
  setEnabled,
  constraints,
  onChange,
}: {
  enabled: boolean;
  setEnabled: (v: boolean) => void;
  constraints: GAConstraints;
  onChange: (c: GAConstraints) => void;
}) {
  function update(patch: Partial<GAConstraints>) {
    onChange({ ...constraints, ...patch });
  }

  return (
    <div className="bg-slate-800 rounded-xl p-5 shadow">
      <label className="flex items-center gap-2 mb-3">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => setEnabled(e.target.checked)}
        />
        <span className="text-lg font-semibold">Apply Constraint-Aware GA filter</span>
      </label>
      <p className="text-sm text-slate-400 mb-4">
        After ranking, select the optimal subset under budget, diversity and
        minimum-fit constraints (Malini et al. 2026).
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <Field label="Total budget (USD)">
          <input
            type="number"
            min={0}
            disabled={!enabled}
            value={constraints.budget}
            onChange={(e) => update({ budget: Number(e.target.value) })}
            className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 disabled:opacity-50"
          />
        </Field>
        <Field label="Min female ratio (0-1)">
          <input
            type="number"
            step={0.05}
            min={0}
            max={1}
            disabled={!enabled}
            value={constraints.min_female_ratio}
            onChange={(e) => update({ min_female_ratio: Number(e.target.value) })}
            className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 disabled:opacity-50"
          />
        </Field>
        <Field label="Min fit threshold (0-1)">
          <input
            type="number"
            step={0.05}
            min={0}
            max={1}
            disabled={!enabled}
            value={constraints.min_fit_threshold}
            onChange={(e) => update({ min_fit_threshold: Number(e.target.value) })}
            className="w-full bg-slate-900 border border-slate-700 rounded px-2 py-1 disabled:opacity-50"
          />
        </Field>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block text-sm text-slate-300">
      <span className="block mb-1">{label}</span>
      {children}
    </label>
  );
}
