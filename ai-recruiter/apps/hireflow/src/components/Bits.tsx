import { motion } from "framer-motion";

export function FitBadge({ value, size = "md" }: { value: number; size?: "sm" | "md" | "lg" }) {
  const tone =
    value >= 85 ? "from-emerald-400 to-teal-500"
    : value >= 70 ? "from-violet-400 to-fuchsia-500"
    : value >= 55 ? "from-amber-400 to-orange-500"
    : "from-rose-400 to-red-500";
  const dim = size === "sm" ? 32 : size === "lg" ? 56 : 44;
  const stroke = size === "sm" ? 2.5 : size === "lg" ? 3.5 : 3;
  const r = (dim - stroke) / 2;
  const c = 2 * Math.PI * r;
  return (
    <div className="relative grid place-items-center" style={{ width: dim, height: dim }}>
      <svg width={dim} height={dim} className="-rotate-90">
        <circle cx={dim/2} cy={dim/2} r={r} fill="none" stroke="currentColor" strokeOpacity={0.12} strokeWidth={stroke} />
        <motion.circle
          cx={dim/2} cy={dim/2} r={r} fill="none"
          strokeWidth={stroke} strokeLinecap="round"
          stroke="url(#fit-grad)"
          initial={{ strokeDasharray: `0 ${c}` }}
          animate={{ strokeDasharray: `${(value/100)*c} ${c}` }}
          transition={{ duration: 0.9, ease: "easeOut" }}
        />
        <defs>
          <linearGradient id="fit-grad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="oklch(0.7 0.2 295)" />
            <stop offset="100%" stopColor="oklch(0.72 0.16 155)" />
          </linearGradient>
        </defs>
      </svg>
      <div className={`absolute font-mono font-semibold ${size === "sm" ? "text-[9px]" : size === "lg" ? "text-base" : "text-xs"}`}>
        {value}
      </div>
    </div>
  );
}

export function StageChip({ stage }: { stage: string }) {
  const styles: Record<string, string> = {
    New: "bg-muted text-foreground",
    Screen: "bg-warning/15 text-warning-foreground",
    Interview: "bg-accent/15 text-accent",
    Offer: "bg-success/20 text-success-foreground",
    Rejected: "bg-destructive/15 text-destructive",
  };
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-medium ${styles[stage] || ""}`}>
      {stage}
    </span>
  );
}

export function Sparkline({ data, width = 80, height = 24 }: { data: number[]; width?: number; height?: number }) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const step = width / (data.length - 1);
  const points = data
    .map((v, i) => `${i * step},${height - ((v - min) / range) * height}`)
    .join(" ");
  return (
    <svg width={width} height={height}>
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
