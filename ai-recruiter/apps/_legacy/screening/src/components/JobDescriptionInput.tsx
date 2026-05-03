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
    <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-200">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-slate-900">Job Description</h2>
        <label className="text-sm text-indigo-600 cursor-pointer hover:text-indigo-700 font-medium">
          📄 Load .txt
          <input type="file" accept=".txt" className="hidden" onChange={onFile} />
        </label>
      </div>
      <textarea
        className="w-full h-48 p-3 rounded-xl bg-slate-50 border border-slate-200 text-slate-900 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-indigo-200 focus:border-indigo-400 transition-all"
        placeholder="Paste the job description here, including requirements and responsibilities…"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
      {loading && <div className="text-slate-500 text-sm mt-2">Reading file…</div>}
    </div>
  );
}
