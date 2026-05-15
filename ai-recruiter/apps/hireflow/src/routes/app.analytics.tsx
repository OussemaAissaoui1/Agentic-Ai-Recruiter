import { createFileRoute } from "@tanstack/react-router";
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
  Legend,
} from "recharts";
import { motion } from "framer-motion";
import { useAnalytics } from "@/lib/analytics";

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
  const { apps, funnel, trend, sources, team, loading } = useAnalytics();

  const hasApps = apps.length > 0;
  const hasTrend = trend.some((p) => p.applicants > 0);
  const hasTeam = team.length > 0;
  const hasSources = sources.length > 0;
  const sourceTotal = sources.reduce((acc, s) => acc + s.value, 0) || 1;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl tracking-tight">Analytics</h1>
        <p className="mt-1 text-muted-foreground">
          Pipeline health, candidate domain mix, decision velocity — all derived
          from live applications.
        </p>
      </div>

      {loading ? (
        <LoadingSkeleton />
      ) : !hasApps ? (
        <EmptyAnalytics />
      ) : (
        <>
          <div className="grid gap-4 lg:grid-cols-3">
            <Card title="Conversion funnel" className="lg:col-span-2">
              <div className="space-y-2 py-4" role="list" aria-label="Conversion funnel">
                {funnel.map((f, i) => {
                  const max = funnel[0].value || 1;
                  const w = (f.value / max) * 100;
                  const pct = Math.round((f.value / max) * 100);
                  return (
                    <div key={f.stage} role="listitem">
                      <div className="flex justify-between text-xs">
                        <span>{f.stage}</span>
                        <span className="font-mono tabular-nums text-muted-foreground">
                          {f.value}
                          <span className="ml-2 text-muted-foreground/60">{pct}%</span>
                        </span>
                      </div>
                      <div className="mt-1 h-7 overflow-hidden rounded-md bg-muted">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${w}%` }}
                          transition={{ delay: i * 0.08, duration: 0.7 }}
                          className="h-full bg-violet-grad"
                          aria-label={`${f.stage}: ${f.value} candidates`}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>

            <Card
              title="Candidate domains"
              subtitle="Derived from applicant email"
            >
              {hasSources ? (
                <>
                  <div className="h-56">
                    <ResponsiveContainer>
                      <PieChart>
                        <Pie
                          data={sources}
                          dataKey="value"
                          nameKey="name"
                          innerRadius={45}
                          outerRadius={75}
                          paddingAngle={3}
                        >
                          {sources.map((_, i) => (
                            <Cell
                              key={i}
                              fill={COLORS[i % COLORS.length]}
                              stroke="transparent"
                            />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            background: "var(--color-popover)",
                            border: "1px solid var(--color-border)",
                            borderRadius: 12,
                            fontSize: 12,
                          }}
                          formatter={(v: number, name: string) => [
                            `${v} (${Math.round((v / sourceTotal) * 100)}%)`,
                            name,
                          ]}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <ul className="space-y-1 text-xs">
                    {sources.map((s, i) => {
                      const pct = Math.round((s.value / sourceTotal) * 100);
                      return (
                        <li
                          key={s.name}
                          className="flex items-center justify-between"
                        >
                          <span className="inline-flex items-center gap-2">
                            <span
                              className="h-2 w-2 rounded-full"
                              style={{ background: COLORS[i % COLORS.length] }}
                              aria-hidden
                            />
                            {s.name}
                          </span>
                          <span className="font-mono tabular-nums text-muted-foreground">
                            {s.value} · {pct}%
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                </>
              ) : (
                <EmptyState
                  title="No domain signal yet"
                  hint="Once applications come in with emails this will populate."
                />
              )}
            </Card>
          </div>

          <Card title="Applicants vs average fit (14d)">
            {hasTrend ? (
              <div className="h-72">
                <ResponsiveContainer>
                  <AreaChart data={trend}>
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
                    <XAxis
                      dataKey="day"
                      tick={{ fontSize: 11, fill: "currentColor", opacity: 0.6 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{ fontSize: 11, fill: "currentColor", opacity: 0.5 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--color-popover)",
                        border: "1px solid var(--color-border)",
                        borderRadius: 12,
                        fontSize: 12,
                      }}
                    />
                    <Legend
                      verticalAlign="top"
                      height={28}
                      iconType="circle"
                      wrapperStyle={{ fontSize: 12 }}
                    />
                    <Area
                      type="monotone"
                      dataKey="applicants"
                      name="Applicants"
                      stroke="oklch(0.7 0.2 295)"
                      strokeWidth={2}
                      fill="url(#a1)"
                    />
                    <Area
                      type="monotone"
                      dataKey="quality"
                      name="Avg fit"
                      stroke="oklch(0.72 0.16 155)"
                      strokeWidth={2}
                      fill="url(#a2)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState
                title="No applications in the last 14 days"
                hint="As candidates apply, the trend will fill in."
              />
            )}
          </Card>

          <Card title="Open vs hired by team">
            {hasTeam ? (
              <div className="h-72">
                <ResponsiveContainer>
                  <BarChart data={team}>
                    <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="team"
                      tick={{ fontSize: 11, fill: "currentColor", opacity: 0.7 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <YAxis
                      allowDecimals={false}
                      tick={{ fontSize: 11, fill: "currentColor", opacity: 0.5 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--color-popover)",
                        border: "1px solid var(--color-border)",
                        borderRadius: 12,
                        fontSize: 12,
                      }}
                    />
                    <Legend
                      verticalAlign="top"
                      height={28}
                      iconType="circle"
                      wrapperStyle={{ fontSize: 12 }}
                    />
                    <Bar dataKey="open" name="Open roles" fill="oklch(0.7 0.2 295)" radius={[8, 8, 0, 0]} />
                    <Bar dataKey="hired" name="Hired" fill="oklch(0.72 0.16 155)" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState
                title="No team data yet"
                hint="Create jobs with a team to compare hiring across teams."
              />
            )}
          </Card>
        </>
      )}
    </div>
  );
}

function Card({
  title,
  subtitle,
  className,
  children,
}: {
  title: string;
  subtitle?: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <section
      className={`rounded-2xl border border-border bg-card p-5 ${className || ""}`}
      aria-label={title}
    >
      <header className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-display text-lg">{title}</h3>
        {subtitle && (
          <span className="text-xs text-muted-foreground">{subtitle}</span>
        )}
      </header>
      <div className="mt-2">{children}</div>
    </section>
  );
}

function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="flex h-56 flex-col items-center justify-center gap-2 rounded-xl bg-muted/30 p-6 text-center">
      <div className="text-sm font-medium">{title}</div>
      {hint && <div className="text-xs text-muted-foreground">{hint}</div>}
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      {Array.from({ length: 3 }).map((_, i) => (
        <div
          key={i}
          className="h-56 animate-pulse rounded-2xl border border-border bg-muted/30"
          aria-hidden
        />
      ))}
    </div>
  );
}

function EmptyAnalytics() {
  return (
    <div className="rounded-3xl border border-dashed border-border bg-card/40 p-12 text-center">
      <h2 className="font-display text-2xl">Analytics will turn on with your first applicant.</h2>
      <p className="mx-auto mt-2 max-w-prose text-sm text-muted-foreground">
        We don't fabricate metrics. The funnel, trend, and team breakdowns are
        computed live from your applications — they'll appear here the moment
        real data exists.
      </p>
    </div>
  );
}
