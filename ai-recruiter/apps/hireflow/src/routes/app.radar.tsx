import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { useMemo, useState } from "react";

import type { Application } from "@/lib/api";
import { useApplications } from "@/lib/queries";

export const Route = createFileRoute("/app/radar")({
  head: () => ({ meta: [{ title: "Talent Radar — HireFlow" }] }),
  component: Radar,
});

// ---------------------------------------------------------------------------
// Layout constants — kept at module scope so the SVG numbers below stay
// self-explanatory ("CENTER + RING_OUTER" reads better than "280 + 240").
// ---------------------------------------------------------------------------
const SIZE = 560;          // svg viewBox edge
const CENTER = SIZE / 2;   // 280
const RING_INNER = 30;     // smallest ring radius (the "core")
const RING_OUTER = 250;    // largest ring radius
const RING_SPAN = RING_OUTER - RING_INNER;

// Stage → color. Maps a candidate's pipeline stage to a stable hue so the
// radar communicates not just "how good a fit" (radial position) but also
// "where in the pipeline" (color).
const STAGE_COLOR: Record<Application["stage"], string> = {
  applied:     "oklch(0.72 0.18 250)", // blue   — new lead
  approved:    "oklch(0.74 0.18 195)", // teal   — moved forward
  interviewed: "oklch(0.74 0.18 165)", // green  — talked to
  offer:       "oklch(0.78 0.18 90)",  // amber  — offer extended
  hired:       "oklch(0.75 0.18 295)", // violet — closed won
  rejected:    "oklch(0.62 0.05 0)",   // gray   — out
};

const STAGE_LABEL: Record<Application["stage"], string> = {
  applied:     "Applied",
  approved:    "Approved",
  interviewed: "Interviewed",
  offer:       "Offer",
  hired:       "Hired",
  rejected:    "Rejected",
};

// Stable hash → angle. Without this the same candidate would jump to a new
// position on every render (which the old `i * 0.07` formula effectively did
// because it depended on array order).
function hashAngle(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return (h % 360) * (Math.PI / 180);
}

function skillDensity(app: Application): number {
  const matched = (app.matched_skills ?? []).length;
  const missing = (app.missing_skills ?? []).length;
  const total = matched + missing;
  return total > 0 ? matched / total : 0;
}

function Radar() {
  const navigate = useNavigate();
  const [hover, setHover] = useState<string | null>(null);
  const { data: apps = [] } = useApplications();

  const points = useMemo(() => {
    return apps.slice(0, 60).map((c) => {
      const fit = Math.max(0, Math.min(1, c.fit_score)); // clamp 0..1
      const density = skillDensity(c);
      // Inverted radius: high fit pulls toward the center. (1 - fit) gives 0
      // at perfect fit (center) and 1 at zero fit (outer ring).
      const radius = RING_INNER + (1 - fit) * RING_SPAN;
      const angle = hashAngle(c.candidate_email || c.id);
      return {
        c,
        fit,
        density,
        x: CENTER + Math.cos(angle) * radius,
        y: CENTER + Math.sin(angle) * radius,
        // Dot size encodes skill density (matched / total skills required).
        // Range: 4–11px so dots stay legible without overwhelming the layout.
        r: 4 + density * 7,
        color: STAGE_COLOR[c.stage] ?? STAGE_COLOR.applied,
      };
    });
  }, [apps]);

  // Pre-compute the ring fit labels. The outer ring is 0% fit, inner ring is
  // 100% — and that's the relationship users have to understand at a glance.
  const ringFits = [1.0, 0.75, 0.5, 0.25];
  const rings = ringFits.map((f) => ({
    fit: f,
    radius: RING_INNER + (1 - f) * RING_SPAN,
  }));

  const hoveredApp = hover ? apps.find((a) => a.id === hover) ?? null : null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-display text-3xl tracking-tight">Talent Radar</h1>
        <p className="mt-1 max-w-2xl text-muted-foreground">
          Each dot is a candidate. <strong className="text-foreground">Distance from center</strong> tracks overall fit
          (closer = stronger). <strong className="text-foreground">Dot size</strong> reflects how many of the
          required skills they have. <strong className="text-foreground">Color</strong> shows their pipeline stage.
        </p>
      </div>

      <div className="rounded-3xl border border-border bg-card p-6 sm:p-8">
        <div className="grid gap-6 lg:grid-cols-[1fr_220px] lg:items-start">
          {/* Radar SVG */}
          <div className="mx-auto w-full" style={{ maxWidth: SIZE }}>
            <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full" aria-label="Talent radar chart">
              {/* Concentric rings + their fit labels on the right axis */}
              {rings.map((ring, i) => (
                <g key={ring.fit}>
                  <circle
                    cx={CENTER}
                    cy={CENTER}
                    r={ring.radius}
                    fill="none"
                    stroke="currentColor"
                    strokeOpacity={i === 0 ? 0.12 : 0.06}
                    strokeDasharray={i === 0 ? "0" : "2 4"}
                  />
                  <text
                    x={CENTER + ring.radius + 6}
                    y={CENTER + 4}
                    fontSize="10"
                    className="fill-muted-foreground/70"
                    fontFamily="var(--font-mono)"
                  >
                    {Math.round(ring.fit * 100)}%
                  </text>
                </g>
              ))}

              {/* Cross-hairs — pure visual scaffolding */}
              {[0, 1, 2, 3].map((i) => {
                const a = (i / 4) * Math.PI * 2;
                return (
                  <line
                    key={i}
                    x1={CENTER + Math.cos(a) * RING_INNER}
                    y1={CENTER + Math.sin(a) * RING_INNER}
                    x2={CENTER + Math.cos(a) * RING_OUTER}
                    y2={CENTER + Math.sin(a) * RING_OUTER}
                    stroke="currentColor"
                    strokeOpacity={0.05}
                  />
                );
              })}

              {/* Center "CORE" marker */}
              <circle cx={CENTER} cy={CENTER} r={RING_INNER - 4} fill="currentColor" fillOpacity={0.03} />
              <text
                x={CENTER}
                y={CENTER + 4}
                textAnchor="middle"
                fontSize="10"
                className="fill-muted-foreground"
                fontFamily="var(--font-mono)"
              >
                CORE
              </text>

              {/* Candidate dots */}
              {points.map((p, i) => {
                const isHover = hover === p.c.id;
                return (
                  <motion.circle
                    key={p.c.id}
                    cx={p.x}
                    cy={p.y}
                    initial={{ r: 0, opacity: 0 }}
                    animate={{ r: isHover ? p.r + 2 : p.r, opacity: 1 }}
                    transition={{ delay: Math.min(i * 0.015, 0.6), type: "spring", stiffness: 220 }}
                    fill={p.color}
                    fillOpacity={isHover ? 1 : 0.78}
                    stroke={isHover ? "currentColor" : "transparent"}
                    strokeOpacity={isHover ? 0.9 : 0}
                    strokeWidth={2}
                    onMouseEnter={() => setHover(p.c.id)}
                    onMouseLeave={() => setHover(null)}
                    onClick={() =>
                      navigate({ to: "/app/applicants/$id/report", params: { id: p.c.id } })
                    }
                    className="cursor-pointer focus:outline-none"
                    role="button"
                    tabIndex={0}
                    aria-label={`${p.c.candidate_name || p.c.candidate_email}, ${Math.round(p.fit * 100)}% fit`}
                  />
                );
              })}
            </svg>
          </div>

          {/* Right rail: legend + hover detail */}
          <aside className="space-y-4">
            <div>
              <h3 className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                Stage
              </h3>
              <ul className="mt-2 space-y-1.5">
                {(Object.keys(STAGE_COLOR) as Application["stage"][]).map((s) => {
                  const count = apps.filter((a) => a.stage === s).length;
                  if (count === 0) return null;
                  return (
                    <li key={s} className="flex items-center gap-2 text-xs">
                      <span
                        aria-hidden
                        className="inline-block h-2.5 w-2.5 rounded-full"
                        style={{ backgroundColor: STAGE_COLOR[s] }}
                      />
                      <span className="text-foreground">{STAGE_LABEL[s]}</span>
                      <span className="ml-auto font-mono text-[11px] tabular-nums text-muted-foreground">
                        {count}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>

            <div className="border-t border-border pt-4">
              <h3 className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                Dot size
              </h3>
              <div className="mt-2 flex items-center gap-3 text-xs text-muted-foreground">
                <svg width="50" height="20" viewBox="0 0 50 20" aria-hidden>
                  <circle cx="8"  cy="10" r="3.5" className="fill-muted-foreground/50" />
                  <circle cx="25" cy="10" r="6"   className="fill-muted-foreground/50" />
                  <circle cx="42" cy="10" r="9"   className="fill-muted-foreground/50" />
                </svg>
                Skill coverage
              </div>
              <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                Matched skills ÷ total required.
              </p>
            </div>

            {hoveredApp && <HoverCard app={hoveredApp} />}
          </aside>
        </div>

        {apps.length === 0 && (
          <div className="mt-6 text-center text-sm text-muted-foreground">
            No candidates yet. Once applications come in, they'll show here.
          </div>
        )}
      </div>
    </div>
  );
}

function HoverCard({ app }: { app: Application }) {
  const matched = (app.matched_skills ?? []).length;
  const missing = (app.missing_skills ?? []).length;
  const fitPct = Math.round(app.fit_score * 100);
  return (
    <div className="border-t border-border pt-4">
      <h3 className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        Selected
      </h3>
      <div className="mt-2 rounded-lg border border-border bg-background p-3 text-sm">
        <div className="break-words font-semibold text-foreground">
          {app.candidate_name || app.candidate_email}
        </div>
        {app.candidate_name && (
          <div className="break-all text-[11px] text-muted-foreground">{app.candidate_email}</div>
        )}
        <dl className="mt-2.5 grid grid-cols-2 gap-y-1 text-[11px]">
          <dt className="text-muted-foreground">Fit</dt>
          <dd className="text-right font-mono tabular-nums text-foreground">{fitPct}%</dd>
          <dt className="text-muted-foreground">Matched</dt>
          <dd className="text-right font-mono tabular-nums text-foreground">{matched}</dd>
          <dt className="text-muted-foreground">Missing</dt>
          <dd className="text-right font-mono tabular-nums text-foreground">{missing}</dd>
          <dt className="text-muted-foreground">Stage</dt>
          <dd className="text-right text-foreground">{STAGE_LABEL[app.stage]}</dd>
        </dl>
        <p className="mt-2 text-[10px] italic text-muted-foreground">Click the dot to open the full report.</p>
      </div>
    </div>
  );
}
