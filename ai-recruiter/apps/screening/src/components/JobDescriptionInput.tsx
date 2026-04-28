import { useState } from "react";

export function JobDescriptionInput({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const [loading, setLoading] = useState(false);

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (!f) return;
    setLoading(true);
    try {
      const text = await f.text();
      onChange(text);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-slate-800 rounded-xl p-5 shadow">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">Job Description</h2>
        <label className="text-sm text-slate-300 cursor-pointer underline">
          Load .txt file
          <input type="file" accept=".txt" className="hidden" onChange={onFile} />
        </label>
      </div>
      <textarea
        className="w-full h-48 p-3 rounded-lg bg-slate-900 border border-slate-700 text-slate-100 font-mono text-sm"
        placeholder="Paste the job description here, including requirements and responsibilities…"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
      {loading && <div className="text-slate-400 text-sm mt-2">Reading file…</div>}
    </div>
  );
}
