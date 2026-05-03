import { createFileRoute } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useMemo, useState } from "react";
import { useApplications } from "@/lib/queries";

export const Route = createFileRoute("/app/radar")({
  head: () => ({ meta: [{ title: "Talent Radar — HireFlow" }] }),
  component: Radar,
});

function avatarHueOf(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h % 360;
}

function Radar() {
  const [hover, setHover] = useState<string | null>(null);
  const { data: apps = [] } = useApplications();

  const points = useMemo(() => {
    return apps.slice(0, 40).map((c, i) => {
      const fit = Math.round(c.fit_score * 100);
      const matched = (c.matched_skills ?? []).length;
      // Skill density approximated by matched skill count.
      const angle = (matched / Math.max(matched, 4)) * Math.PI * 2 + i * 0.07;
      const radius = (fit / 100) * 220;
      const hue = avatarHueOf(c.candidate_email || c.id);
      return {
        c,
        fit,
        hue,
        x: 280 + Math.cos(angle) * radius,
        y: 280 + Math.sin(angle) * radius,
        r: 6 + fit / 14,
      };
    });
  }, [apps]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-4xl tracking-tight">Talent Radar</h1>
        <p className="mt-1 text-muted-foreground">
          Candidates clustered by skill density. Closer to center = higher overall fit.
        </p>
      </div>
      <div className="rounded-3xl border border-border bg-card p-8">
        <div className="mx-auto" style={{ width: 560, maxWidth: "100%" }}>
          <svg viewBox="0 0 560 560" className="w-full">
            {[60, 130, 200, 270].map((r) => (
              <circle key={r} cx="280" cy="280" r={r} fill="none" stroke="currentColor" strokeOpacity={0.06} />
            ))}
            {[0, 1, 2, 3].map((i) => {
              const a = (i / 4) * Math.PI * 2;
              return (
                <line
                  key={i}
                  x1="280"
                  y1="280"
                  x2={280 + Math.cos(a) * 270}
                  y2={280 + Math.sin(a) * 270}
                  stroke="currentColor"
                  strokeOpacity={0.06}
                />
              );
            })}
            {points.map((p, i) => (
              <motion.circle
                key={p.c.id}
                cx={p.x}
                cy={p.y}
                initial={{ r: 0, opacity: 0 }}
                animate={{ r: p.r, opacity: 1 }}
                transition={{ delay: i * 0.02, type: "spring", stiffness: 200 }}
                fill={`oklch(0.7 0.18 ${p.hue})`}
                fillOpacity={0.85}
                stroke={hover === p.c.id ? "white" : "transparent"}
                strokeWidth={2}
                onMouseEnter={() => setHover(p.c.id)}
                onMouseLeave={() => setHover(null)}
                className="cursor-pointer transition"
              />
            ))}
            <text x="280" y="284" textAnchor="middle" className="fill-muted-foreground" fontSize="10" fontFamily="var(--font-mono)">
              CORE
            </text>
          </svg>
        </div>
        {hover &&
          (() => {
            const c = apps.find((x) => x.id === hover);
            if (!c) return null;
            return (
              <div className="mx-auto mt-4 max-w-md rounded-xl border border-border bg-background p-3 text-sm">
                <div className="font-semibold">
                  {c.candidate_name || c.candidate_email}
                </div>
                <div className="text-xs text-muted-foreground">
                  {c.candidate_email} · fit {Math.round(c.fit_score * 100)}
                </div>
              </div>
            );
          })()}
        {apps.length === 0 && (
          <div className="mt-6 text-center text-sm text-muted-foreground">No candidates yet.</div>
        )}
      </div>
    </div>
  );
}
