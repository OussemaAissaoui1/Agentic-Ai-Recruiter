import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import { ArrowRight, Sparkles, BarChart3, Users, Bot, Shield, Zap } from "lucide-react";
import { useApp } from "@/lib/store";
import { useEffect } from "react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "HireFlow — AI Recruiting that thinks like your best hiring manager" },
      { name: "description", content: "End-to-end AI recruiting platform. Score candidates fairly, run async AI interviews, and make hiring feel inevitable." },
      { property: "og:title", content: "HireFlow — AI Recruiting Platform" },
      { property: "og:description", content: "Score, interview, and decide in one calm workspace." },
    ],
  }),
  component: Landing,
});

function Landing() {
  const { theme } = useApp();
  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.classList.toggle("dark", theme === "dark");
    }
  }, [theme]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-background text-foreground">
      <div className="absolute inset-0 -z-10 bg-aurora" />
      <div className="absolute inset-0 -z-10 grid-fade" />

      {/* Nav */}
      <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-6">
        <Link to="/" className="flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-xl bg-ink">
            <span className="font-display text-base text-white">H</span>
          </div>
          <span className="font-display text-xl tracking-tight">HireFlow</span>
        </Link>
        <nav className="hidden items-center gap-7 text-sm text-muted-foreground md:flex">
          <a href="#product" className="hover:text-foreground">Product</a>
          <a href="#workflow" className="hover:text-foreground">Workflow</a>
          <a href="#trust" className="hover:text-foreground">Trust</a>
        </nav>
        <div className="flex items-center gap-2">
          <Link to="/c" className="hidden rounded-full border border-border px-4 py-2 text-sm font-medium hover:bg-muted md:inline-block">
            For candidates
          </Link>
          <Link
            to="/app"
            className="inline-flex items-center gap-1.5 rounded-full bg-foreground px-4 py-2 text-sm font-medium text-background hover:opacity-90"
          >
            Open recruiter app <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="mx-auto max-w-6xl px-6 pt-12 pb-24 text-center md:pt-20">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mx-auto inline-flex items-center gap-2 rounded-full border border-border bg-card/60 px-3 py-1 text-xs text-muted-foreground backdrop-blur"
        >
          <span className="grid h-1.5 w-1.5 place-items-center rounded-full bg-success" />
          Now with AI Interview Room — async, structured, fair
        </motion.div>
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.05 }}
          className="font-display mx-auto mt-6 max-w-4xl text-balance text-5xl leading-[1.05] tracking-tight md:text-7xl"
        >
          Hiring, finally as <em className="italic text-accent">deliberate</em> as the people you want to hire.
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1 }}
          className="mx-auto mt-6 max-w-xl text-pretty text-base text-muted-foreground md:text-lg"
        >
          HireFlow scores candidates with explainable AI, runs structured async interviews, and gives recruiters a calm room to make great calls — fast.
        </motion.p>
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.15 }}
          className="mt-9 flex items-center justify-center gap-3"
        >
          <Link to="/app" className="group inline-flex items-center gap-1.5 rounded-full bg-violet-grad px-5 py-2.5 text-sm font-semibold text-accent-foreground shadow-glow transition hover:scale-[1.02]">
            Try the recruiter app <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
          </Link>
          <Link to="/c" className="rounded-full border border-border bg-card/60 px-5 py-2.5 text-sm font-semibold backdrop-blur hover:bg-card">
            I'm a candidate
          </Link>
        </motion.div>

        {/* Hero preview card */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.25 }}
          className="relative mx-auto mt-16 max-w-5xl"
        >
          <div className="absolute -inset-2 rounded-3xl bg-violet-grad opacity-20 blur-2xl" />
          <div className="relative overflow-hidden rounded-3xl border border-border bg-card shadow-card-soft">
            <div className="flex items-center gap-1.5 border-b border-border bg-muted/40 px-4 py-3">
              <div className="h-2.5 w-2.5 rounded-full bg-destructive/60" />
              <div className="h-2.5 w-2.5 rounded-full bg-warning/60" />
              <div className="h-2.5 w-2.5 rounded-full bg-success/60" />
              <div className="ml-3 font-mono text-[11px] text-muted-foreground">hireflow.app/applicants</div>
            </div>
            <div className="grid gap-4 p-6 md:grid-cols-3">
              {[
                { name: "Aria Park", role: "Senior Frontend Eng", fit: 92 },
                { name: "Diego Costa", role: "Staff Designer", fit: 81 },
                { name: "Maya Levi", role: "ML Engineer", fit: 74 },
              ].map((c, i) => (
                <motion.div
                  key={c.name}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 + i * 0.1 }}
                  className="rounded-2xl border border-border bg-background p-4 text-left"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-semibold">{c.name}</div>
                      <div className="text-xs text-muted-foreground">{c.role}</div>
                    </div>
                    <div className="font-mono text-2xl font-semibold text-accent">{c.fit}</div>
                  </div>
                  <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${c.fit}%` }}
                      transition={{ duration: 1, delay: 0.5 + i * 0.1 }}
                      className="h-full bg-violet-grad"
                    />
                  </div>
                  <div className="mt-3 flex flex-wrap gap-1">
                    {["React","TS","GraphQL"].map((s) => (
                      <span key={s} className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">{s}</span>
                    ))}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </section>

      {/* Feature grid */}
      <section id="product" className="mx-auto max-w-6xl px-6 py-24">
        <div className="mb-12 max-w-2xl">
          <div className="text-xs uppercase tracking-widest text-muted-foreground">The platform</div>
          <h2 className="font-display mt-2 text-4xl tracking-tight md:text-5xl">One workspace. Two sides. Zero friction.</h2>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: i * 0.05 }}
              className="group relative overflow-hidden rounded-2xl border border-border bg-card p-6 transition hover:shadow-card-soft"
            >
              <div className="mb-4 inline-grid h-10 w-10 place-items-center rounded-xl bg-violet-grad text-white">
                <f.icon className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-semibold">{f.title}</h3>
              <p className="mt-2 text-sm text-muted-foreground">{f.body}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Workflow */}
      <section id="workflow" className="mx-auto max-w-6xl px-6 py-24">
        <div className="grid gap-12 md:grid-cols-2 md:items-center">
          <div>
            <div className="text-xs uppercase tracking-widest text-muted-foreground">Workflow</div>
            <h2 className="font-display mt-2 text-4xl tracking-tight md:text-5xl">From first signal to signed offer.</h2>
            <ol className="mt-8 space-y-5">
              {STEPS.map((s, i) => (
                <li key={s.title} className="flex gap-4">
                  <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full border border-border bg-card font-mono text-xs">
                    {i + 1}
                  </div>
                  <div>
                    <div className="font-semibold">{s.title}</div>
                    <div className="text-sm text-muted-foreground">{s.body}</div>
                  </div>
                </li>
              ))}
            </ol>
          </div>
          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="relative aspect-[4/5] overflow-hidden rounded-3xl border border-border bg-ink p-8 text-white shadow-card-soft"
          >
            <div className="absolute inset-0 bg-violet-grad opacity-30" />
            <div className="relative">
              <div className="text-xs uppercase tracking-widest text-white/60">AI Interview Room</div>
              <div className="font-display mt-2 text-3xl">"Walk me through a system you designed end-to-end."</div>
              <div className="mt-8 flex items-center gap-3">
                <div className="grid h-12 w-12 place-items-center rounded-full bg-white/15 backdrop-blur">
                  <Bot className="h-5 w-5" />
                </div>
                <div className="flex flex-1 items-center gap-1">
                  {Array.from({ length: 28 }).map((_, i) => (
                    <motion.span
                      key={i}
                      animate={{ height: [6, 18 + Math.sin(i) * 10, 8] }}
                      transition={{ duration: 1.4, repeat: Infinity, delay: i * 0.05 }}
                      className="w-1 rounded bg-white/70"
                    />
                  ))}
                </div>
              </div>
              <div className="mt-8 grid grid-cols-3 gap-3 text-xs">
                {[
                  { l: "Clarity", v: 86 },
                  { l: "Depth", v: 78 },
                  { l: "Pace", v: 92 },
                ].map((m) => (
                  <div key={m.l} className="rounded-xl bg-white/10 p-3 backdrop-blur">
                    <div className="text-white/70">{m.l}</div>
                    <div className="font-mono text-lg">{m.v}</div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Trust */}
      <section id="trust" className="mx-auto max-w-6xl px-6 py-24">
        <div className="rounded-3xl border border-border bg-card p-10 md:p-14">
          <div className="grid gap-10 md:grid-cols-2 md:items-center">
            <div>
              <div className="text-xs uppercase tracking-widest text-muted-foreground">Built for trust</div>
              <h2 className="font-display mt-2 text-3xl tracking-tight md:text-4xl">Explainable, auditable, fair by default.</h2>
              <p className="mt-4 text-pretty text-muted-foreground">
                Every score breaks down into evidence. Every AI decision is reviewable. Bias checks run on every shortlist — not as a quarterly report.
              </p>
              <Link to="/app" className="mt-6 inline-flex items-center gap-1.5 rounded-full bg-foreground px-5 py-2.5 text-sm font-semibold text-background">
                See it in the app <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm">
              {[
                { k: "GDPR", v: "Compliant by design" },
                { k: "SOC 2", v: "Type II in progress" },
                { k: "Bias audit", v: "Per-shortlist, per-job" },
                { k: "Data", v: "Your tenant. Your keys." },
              ].map((t) => (
                <div key={t.k} className="rounded-2xl border border-border bg-background p-4">
                  <div className="text-xs text-muted-foreground">{t.k}</div>
                  <div className="mt-1 font-medium">{t.v}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t border-border">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-6 py-8 text-sm text-muted-foreground md:flex-row">
          <div>© 2026 HireFlow. Hiring with intention.</div>
          <div className="flex gap-5">
            <Link to="/app" className="hover:text-foreground">Recruiter app</Link>
            <Link to="/c" className="hover:text-foreground">Candidate app</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}

const FEATURES = [
  { icon: Sparkles, title: "AI Job Wizard", body: "Type a few notes. Get a polished, on-brand JD with calibrated must-haves and nice-to-haves — streaming as you watch." },
  { icon: Users, title: "Explainable Ranking", body: "Fit scores broken down into skills, trajectory, and culture — with the exact lines from the CV that earned each point." },
  { icon: Bot, title: "AI Interview Room", body: "Async structured interviews with a calm AI host. Behavioral signals + transcript + score, in one quiet room." },
  { icon: BarChart3, title: "Pipeline Analytics", body: "Funnel velocity, source quality, time-to-decision. The metrics your hiring committee actually asks for." },
  { icon: Zap, title: "Decision Queue", body: "Tinder-fast triage of new applicants with keyboard shortcuts. Never wonder who's waiting on you." },
  { icon: Shield, title: "Bias Watchdog", body: "Per-shortlist parity checks across gender, geography, and seniority — with one-click rebalance suggestions." },
];

const STEPS = [
  { title: "Open the role", body: "Streaming AI JD wizard turns notes into a calibrated job description in 30 seconds." },
  { title: "Let the room fill", body: "Candidates apply or get sourced. Fit scores appear with full evidence trails." },
  { title: "Run structured interviews", body: "Async AI room asks the same questions, captures the same signals, scores fairly." },
  { title: "Decide together", body: "Compare top 3 side-by-side, see bias checks, send the offer with one click." },
];
