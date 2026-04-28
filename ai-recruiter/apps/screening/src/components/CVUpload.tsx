import { useRef } from "react";

export function CVUpload({
  files,
  onChange,
}: {
  files: File[];
  onChange: (files: File[]) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  function add(incoming: FileList | null) {
    if (!incoming) return;
    const existing = new Set(files.map((f) => f.name + "|" + f.size));
    const merged = [...files];
    for (const f of Array.from(incoming)) {
      const key = f.name + "|" + f.size;
      if (!existing.has(key)) {
        merged.push(f);
        existing.add(key);
      }
    }
    onChange(merged);
  }

  function remove(i: number) {
    onChange(files.filter((_, idx) => idx !== i));
  }

  return (
    <div className="bg-slate-800 rounded-xl p-5 shadow">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">Resumes / CVs</h2>
        <span className="text-sm text-slate-400">{files.length} uploaded</span>
      </div>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          add(e.dataTransfer.files);
        }}
        className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center text-slate-400 cursor-pointer hover:bg-slate-900"
      >
        <div className="text-3xl mb-1">+</div>
        <div>Click or drop CVs here</div>
        <div className="text-xs mt-1">PDF, DOCX, TXT, or images (PNG/JPG)</div>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.doc,.txt,.png,.jpg,.jpeg,.bmp,.tiff,.tif,.webp"
          className="hidden"
          onChange={(e) => add(e.target.files)}
        />
      </div>
      {files.length > 0 && (
        <ul className="mt-4 space-y-2">
          {files.map((f, i) => (
            <li
              key={i}
              className="flex items-center justify-between bg-slate-900 rounded-lg px-3 py-2 text-sm"
            >
              <span className="truncate pr-2">{f.name}</span>
              <button
                onClick={() => remove(i)}
                className="text-red-400 hover:text-red-300"
              >
                remove
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
