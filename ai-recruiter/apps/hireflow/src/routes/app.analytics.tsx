import { createFileRoute } from "@tanstack/react-router";
import { ANALYTICS } from "@/lib/mock";
import {
  BarChart,
  Bar,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
  CartesianGrid,
} from "recharts";
import { motion } from "framer-motion";
import { useApplications } from "@/lib/queries";
import { useMemo } from "react";

export const Route = createFileRoute("/app/analytics")({
  head: () => ({ meta: [{ title: "Analytics — HireFlow" }] }),
  component: Analytics,
});

const COLORS = [
  "oklch(0.7 0.2 295)",
  "oklch(0.72 0.16 155)",
  "oklch(0.8 0.16 80)",
  "oklch(0.65 0.22 27)",
];

function Analytics() {
  const { data: apps = [] } = useApplications();

  const funnel = useMemo(() => {
    if (apps.length === 0) return ANALYTICS.funnel;
    const counts = {
      Sourced: apps.length,
      Applied: apps.length,
      Screened: apps.filter((a) =>
        ["screened", "interview", "interviewed", "offer", "hired"].includes(a.stage),
      ).length,
      Interviewed: apps.filter((a) =>
        ["interview", "interviewed", "offer", "hired"].includes(a.stage),
      ).length,
      Offered: apps.filter((a) => ["offer", "hired"].includes(a.stage)).length,
      Hired: apps.filter((a) => a.stage === "hired").length,
    };
    return Object.entries(counts).map(([stage, value]) => ({ stage, value }));
  }, [apps]);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-4xl tracking-tight">Analytics</h1>
        <p className="mt-1 text-muted-foreground">Pipeline health, source quality, decision velocity.</p>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="Conversion funnel" className="lg:col-span-2">
          <div className="space-y-2 py-4">
            {funnel.map((f, i) => {
              const max = funnel[0].value || 1;
              const w = (f.value / max) * 100;
              return (
                <div key={f.stage}>
                  <div className="flex justify-between text-xs">
                    <span>{f.stage}</span>
                    <span className="font-mono text-muted-foreground">{f.value}</span>
                  </div>
                  <div className="mt-1 h-7 overflow-hidden rounded-md bg-muted">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${w}%` }}
                      transition={{ delay: i * 0.08, duration: 0.7 }}
                      className="h-full bg-violet-grad"
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </Card>

        <Card title="Sources">
          <div className="h-56">
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={ANALYTICS.sources}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={45}
                  outerRadius={75}
                  paddingAngle={3}
                >
                  {ANALYTICS.sources.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} stroke="transparent" />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "var(--color-popover)",
                    border: "1px solid var(--color-border)",
                    borderRadius: 12,
                    fontSize: 12,
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-1 text-xs">
            {ANALYTICS.sources.map((s, i) => (
              <div key={s.name} className="flex items-center justify-between">
                <span className="inline-flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full" style={{ background: COLORS[i % COLORS.length] }} />
                  {s.name}
                </span>
                <span className="font-mono text-muted-foreground">{s.value}%</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Card title="Applicants vs quality (14d)">
        <div className="h-72">
          <ResponsiveContainer>
            <AreaChart data={ANALYTICS.trend}>
              <defs>
                <linearGradient id="a1" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="oklch(0.7 0.2 295)" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="oklch(0.7 0.2 295)" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="a2" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="oklch(0.72 0.16 155)" stopOpacity={0.4} />
                  <stop offset="100%" stopColor="oklch(0.72 0.16 155)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="day" tick={{ fontSize: 11, fill: "currentColor", opacity: 0.5 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: "currentColor", opacity: 0.5 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{
                  background: "var(--color-popover)",
                  border: "1px solid var(--color-border)",
                  borderRadius: 12,
                  fontSize: 12,
                }}
              />
              <Area type="monotone" dataKey="applicants" stroke="oklch(0.7 0.2 295)" strokeWidth={2} fill="url(#a1)" />
              <Area type="monotone" dataKey="quality" stroke="oklch(0.72 0.16 155)" strokeWidth={2} fill="url(#a2)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card title="Open vs hired by team">
        <div className="h-72">
          <ResponsiveContainer>
            <BarChart data={ANALYTICS.pipeline}>
              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="team" tick={{ fontSize: 11, fill: "currentColor", opacity: 0.6 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: "currentColor", opacity: 0.5 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{
                  background: "var(--color-popover)",
                  border: "1px solid var(--color-border)",
                  borderRadius: 12,
                  fontSize: 12,
                }}
              />
              <Bar dataKey="open" fill="oklch(0.7 0.2 295)" radius={[8, 8, 0, 0]} />
              <Bar dataKey="hired" fill="oklch(0.72 0.16 155)" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
}

function Card({ title, className, children }: { title: string; className?: string; children: React.ReactNode }) {
  return (
    <div className={`rounded-2xl border border-border bg-card p-5 ${className || ""}`}>
      <h3 className="font-display text-lg">{title}</h3>
      <div className="mt-2">{children}</div>
    </div>
  );
}
