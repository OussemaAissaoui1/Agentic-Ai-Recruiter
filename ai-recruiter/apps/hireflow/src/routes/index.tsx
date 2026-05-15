import { createFileRoute, Link } from "@tanstack/react-router";
import {
  AnimatePresence,
  motion,
  useReducedMotion,
  useScroll,
  useTransform,
  type Variants,
} from "framer-motion";
import {
  ArrowRight,
  BarChart3,
  Bot,
  CheckCircle2,
  Menu,
  Quote,
  Shield,
  Sparkles,
  Users,
  X,
  Zap,
} from "lucide-react";
import { useApp } from "@/lib/store";
import { useEffect, useRef, useState } from "react";
import { AnimatedNumber } from "@/components/motion/AnimatedNumber";
import { SpotlightCard } from "@/components/motion/SpotlightCard";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "HireFlow — AI Recruiting that thinks like your best hiring manager" },
      {
        name: "description",
        content:
          "End-to-end AI recruiting platform. Score candidates fairly, run async AI interviews, and make hiring feel inevitable.",
      },
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
      <BackgroundLayers />
      <ScrollHeader />

      <Hero />

      <StatsBand />

      <FeaturesBento />

      <Workflow />

      <Testimonial />

      <FinalCTA />

      <Footer />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Background — aurora + grid + slow-drifting glows
// ─────────────────────────────────────────────────────────────────────────────

function BackgroundLayers() {
  return (
    <>
      <div className="absolute inset-0 -z-10 bg-aurora" />
      <div className="absolute inset-0 -z-10 grid-fade" />
      {/* Slow-drifting violet/green orbs add depth without competing for attention. */}
      <motion.div
        aria-hidden
        animate={{ x: [-40, 30, -40], y: [-20, 30, -20] }}
        transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
        className="pointer-events-none absolute -top-32 left-1/4 -z-10 h-[420px] w-[420px] rounded-full bg-violet-grad opacity-30 blur-3xl"
      />
      <motion.div
        aria-hidden
        animate={{ x: [30, -30, 30], y: [10, -25, 10] }}
        transition={{ duration: 22, repeat: Infinity, ease: "easeInOut" }}
        className="pointer-events-none absolute top-[40%] right-[10%] -z-10 h-[380px] w-[380px] rounded-full bg-success/30 opacity-30 blur-3xl"
      />
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Header — shrinks, blurs, and adds a hairline divider once you scroll
// ─────────────────────────────────────────────────────────────────────────────

const NAV_ITEMS = [
  { id: "product", label: "Product" },
  { id: "workflow", label: "Workflow" },
  { id: "trust", label: "Trust" },
] as const;

const HEADER_OFFSET = 80;

function ScrollHeader() {
  const reducedMotion = useReducedMotion();
  const [scrolled, setScrolled] = useState(false);
  const [active, setActive] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const firstMobileLinkRef = useRef<HTMLAnchorElement>(null);

  // Shrink header on scroll
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Scroll-spy: highlight the section currently in the reader's middle band.
  useEffect(() => {
    const targets = NAV_ITEMS
      .map((i) => document.getElementById(i.id))
      .filter((el): el is HTMLElement => !!el);
    if (targets.length === 0) return;
    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (visible) setActive(visible.target.id);
      },
      { rootMargin: "-35% 0px -55% 0px", threshold: [0, 0.25, 0.5, 0.75, 1] }
    );
    targets.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);

  // Mobile-menu side effects: body-scroll lock + Esc-to-close + focus mgmt
  useEffect(() => {
    if (!menuOpen) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMenuOpen(false);
    };
    window.addEventListener("keydown", onKey);
    const t = window.setTimeout(() => firstMobileLinkRef.current?.focus(), 30);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
      window.clearTimeout(t);
    };
  }, [menuOpen]);

  // Close menu on resize past md so state stays in sync with layout
  useEffect(() => {
    const onResize = () => {
      if (window.innerWidth >= 768) setMenuOpen(false);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const goToSection = (id: string) => {
    const el = document.getElementById(id);
    if (!el) return;
    const top = el.getBoundingClientRect().top + window.scrollY - HEADER_OFFSET;
    window.scrollTo({ top, behavior: reducedMotion ? "auto" : "smooth" });
    if (typeof history !== "undefined") {
      history.replaceState(null, "", `#${id}`);
    }
    setActive(id);
    setMenuOpen(false);
  };

  return (
    <motion.header
      initial={false}
      animate={{
        paddingTop: scrolled ? 10 : 20,
        paddingBottom: scrolled ? 10 : 20,
      }}
      transition={reducedMotion ? { duration: 0 } : { duration: 0.25, ease: [0.2, 0.8, 0.2, 1] }}
      className={`sticky top-0 z-40 transition-colors ${
        scrolled
          ? "border-b border-border bg-background/80 supports-[backdrop-filter]:bg-background/60 backdrop-blur-xl"
          : "border-b border-transparent bg-transparent"
      }`}
    >
      <div className="relative mx-auto flex max-w-6xl items-center justify-between gap-3 px-4 sm:px-6">
        <Link
          to="/"
          aria-label="HireFlow home"
          className="-m-1 flex items-center gap-2 rounded-md p-1 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background"
        >
          <img
            src="/logo-mark.png"
            alt=""
            width={32}
            height={32}
            className="h-8 w-8 select-none object-contain"
            draggable={false}
          />
          <span className="font-display text-xl tracking-tight">HireFlow</span>
        </Link>

        {/* Desktop nav */}
        <nav aria-label="Primary" className="hidden md:block">
          <ul className="flex items-center gap-1">
            {NAV_ITEMS.map((item) => {
              const isActive = active === item.id;
              return (
                <li key={item.id}>
                  <a
                    href={`#${item.id}`}
                    onClick={(e) => {
                      e.preventDefault();
                      goToSection(item.id);
                    }}
                    aria-current={isActive ? "true" : undefined}
                    className={`relative flex min-h-[40px] items-center rounded-md px-3 py-2 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                      isActive
                        ? "text-foreground"
                        : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
                    }`}
                  >
                    {item.label}
                    {isActive &&
                      (reducedMotion ? (
                        <span
                          aria-hidden
                          className="pointer-events-none absolute inset-x-3 -bottom-0.5 h-0.5 rounded-full bg-violet-grad"
                        />
                      ) : (
                        <motion.span
                          layoutId="nav-indicator"
                          aria-hidden
                          className="pointer-events-none absolute inset-x-3 -bottom-0.5 h-0.5 rounded-full bg-violet-grad"
                          transition={{ type: "spring", stiffness: 380, damping: 32 }}
                        />
                      ))}
                  </a>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Right cluster */}
        <div className="flex items-center gap-1.5">
          <Link
            to="/c"
            className="hidden min-h-[40px] items-center rounded-full border border-border px-4 text-sm font-medium transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background md:inline-flex"
          >
            For candidates
          </Link>
          <Link
            to="/app"
            className="press-tight inline-flex min-h-[44px] items-center gap-1.5 rounded-full bg-foreground px-4 text-sm font-medium text-background transition hover:scale-[1.02] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          >
            <span className="hidden sm:inline">Open recruiter app</span>
            <span className="sm:hidden">Open app</span>
            <ArrowRight className="h-3.5 w-3.5" aria-hidden />
          </Link>
          <button
            type="button"
            onClick={() => setMenuOpen((v) => !v)}
            aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
            aria-expanded={menuOpen}
            aria-controls="mobile-nav-menu"
            className="grid h-11 w-11 place-items-center rounded-md text-foreground transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent md:hidden"
          >
            <AnimatePresence initial={false} mode="wait">
              {menuOpen ? (
                <motion.span
                  key="x"
                  initial={reducedMotion ? false : { rotate: -90, opacity: 0 }}
                  animate={reducedMotion ? { opacity: 1 } : { rotate: 0, opacity: 1 }}
                  exit={reducedMotion ? { opacity: 0 } : { rotate: 90, opacity: 0 }}
                  transition={{ duration: 0.15 }}
                  className="grid place-items-center"
                >
                  <X className="h-5 w-5" aria-hidden />
                </motion.span>
              ) : (
                <motion.span
                  key="m"
                  initial={reducedMotion ? false : { rotate: 90, opacity: 0 }}
                  animate={reducedMotion ? { opacity: 1 } : { rotate: 0, opacity: 1 }}
                  exit={reducedMotion ? { opacity: 0 } : { rotate: -90, opacity: 0 }}
                  transition={{ duration: 0.15 }}
                  className="grid place-items-center"
                >
                  <Menu className="h-5 w-5" aria-hidden />
                </motion.span>
              )}
            </AnimatePresence>
          </button>
        </div>
      </div>

      {/* Mobile menu — dialog beneath the header */}
      <AnimatePresence>
        {menuOpen && (
          <>
            <motion.div
              key="backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
              onClick={() => setMenuOpen(false)}
              className="fixed inset-0 -z-10 bg-foreground/40 backdrop-blur-sm md:hidden"
              aria-hidden
            />
            <motion.div
              key="panel"
              id="mobile-nav-menu"
              role="dialog"
              aria-modal="true"
              aria-label="Site navigation"
              initial={reducedMotion ? { opacity: 0 } : { opacity: 0, y: -8 }}
              animate={reducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
              exit={reducedMotion ? { opacity: 0 } : { opacity: 0, y: -8 }}
              transition={{ duration: 0.2, ease: [0.2, 0.8, 0.2, 1] }}
              className="absolute inset-x-0 top-full border-b border-border bg-background/95 shadow-card-soft backdrop-blur-xl md:hidden"
            >
              <ul className="mx-auto flex max-w-6xl flex-col gap-1 px-3 py-3">
                {NAV_ITEMS.map((item, idx) => {
                  const isActive = active === item.id;
                  return (
                    <li key={item.id}>
                      <a
                        ref={idx === 0 ? firstMobileLinkRef : undefined}
                        href={`#${item.id}`}
                        onClick={(e) => {
                          e.preventDefault();
                          goToSection(item.id);
                        }}
                        aria-current={isActive ? "true" : undefined}
                        className={`flex min-h-[48px] items-center rounded-lg px-3 text-base font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                          isActive
                            ? "bg-accent/10 text-foreground"
                            : "text-foreground hover:bg-muted"
                        }`}
                      >
                        {item.label}
                        {isActive && (
                          <span aria-hidden className="ml-auto h-1.5 w-1.5 rounded-full bg-violet-grad" />
                        )}
                      </a>
                    </li>
                  );
                })}
                <li aria-hidden className="my-1 h-px bg-border" />
                <li>
                  <Link
                    to="/c"
                    onClick={() => setMenuOpen(false)}
                    className="flex min-h-[48px] items-center rounded-lg px-3 text-base font-medium text-foreground transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  >
                    For candidates
                  </Link>
                </li>
                <li className="pt-1">
                  <Link
                    to="/app"
                    onClick={() => setMenuOpen(false)}
                    className="flex min-h-[48px] items-center justify-center gap-1.5 rounded-full bg-violet-grad px-4 text-base font-semibold text-accent-foreground shadow-glow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                  >
                    Open recruiter app
                    <ArrowRight className="h-4 w-4" aria-hidden />
                  </Link>
                </li>
              </ul>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </motion.header>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Hero — word-stagger headline, parallax preview card, live activity ticker
// ─────────────────────────────────────────────────────────────────────────────

const headlineLine1 = "Hiring, finally as".split(" ");
const headlineLine2 = "as the people you want to hire.".split(" ");

const wordVariants: Variants = {
  hidden: { opacity: 0, y: "0.4em", filter: "blur(8px)" },
  show: { opacity: 1, y: "0em", filter: "blur(0px)" },
};

const headlineContainer: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.045, delayChildren: 0.1 } },
};

function Hero() {
  const heroRef = useRef<HTMLDivElement>(null);
  // Parallax: as the page scrolls past the hero, lift the preview card up.
  const { scrollY } = useScroll();
  const cardY = useTransform(scrollY, [0, 600], [0, -60]);
  const cardScale = useTransform(scrollY, [0, 600], [1, 0.97]);

  return (
    <section ref={heroRef} className="mx-auto max-w-6xl px-6 pt-12 pb-24 text-center md:pt-16">
      <motion.h1
        initial="hidden"
        animate="show"
        variants={headlineContainer}
        className="font-display mx-auto flex max-w-4xl flex-wrap items-baseline justify-center gap-x-3 gap-y-1 text-balance text-4xl leading-[1.05] tracking-tight md:text-6xl"
      >
        {headlineLine1.map((w, i) => (
          <motion.span
            key={`a-${i}`}
            variants={wordVariants}
            className="inline-block"
          >
            {w}
          </motion.span>
        ))}
        <motion.em
          variants={wordVariants}
          className="inline-block bg-gradient-to-r from-accent via-accent to-success bg-clip-text italic text-transparent"
        >
          deliberate
        </motion.em>
        {headlineLine2.map((w, i) => (
          <motion.span
            key={`b-${i}`}
            variants={wordVariants}
            className="inline-block"
          >
            {w}
          </motion.span>
        ))}
      </motion.h1>

      <motion.p
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.55 }}
        className="mx-auto mt-6 max-w-xl text-pretty text-base text-muted-foreground md:text-lg"
      >
        HireFlow scores candidates with explainable AI, runs structured async interviews, and gives recruiters a calm room to make great calls — fast.
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.65 }}
        className="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row"
      >
        <Link
          to="/app"
          className="press-tight glow-pulse group inline-flex items-center gap-1.5 rounded-full bg-violet-grad px-5 py-2.5 text-sm font-semibold text-accent-foreground shadow-glow transition-transform hover:scale-[1.04]"
        >
          Try the recruiter app
          <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
        </Link>
        <Link
          to="/c"
          className="press-tight rounded-full border border-border bg-card/60 px-5 py-2.5 text-sm font-semibold backdrop-blur transition-colors hover:bg-card"
        >
          I'm a candidate
        </Link>
      </motion.div>

      <ActivityTicker />

      {/* Hero preview card with subtle scroll-parallax */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.9, delay: 0.4 }}
        style={{ y: cardY, scale: cardScale }}
        className="relative mx-auto mt-14 max-w-5xl"
      >
        <div className="absolute -inset-2 rounded-3xl bg-violet-grad opacity-20 blur-2xl" />
        <SpotlightCard className="relative overflow-hidden rounded-3xl border border-border bg-card shadow-card-soft">
          <div className="flex items-center gap-1.5 border-b border-border bg-muted/40 px-4 py-3">
            <div className="h-2.5 w-2.5 rounded-full bg-destructive/60" aria-hidden />
            <div className="h-2.5 w-2.5 rounded-full bg-warning/60" aria-hidden />
            <div className="h-2.5 w-2.5 rounded-full bg-success/60" aria-hidden />
            <div className="ml-3 font-mono text-[11px] text-muted-foreground">
              hireflow.app/applicants
            </div>
            <span
              className="ml-auto rounded-full border border-border bg-background/70 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-muted-foreground"
              aria-label="Sample data shown for demonstration"
            >
              Sample
            </span>
          </div>
          <div className="grid gap-4 p-6 md:grid-cols-3" aria-label="Sample candidate previews">
            {[
              { name: "Candidate A", role: "Senior Frontend Eng", fit: 92 },
              { name: "Candidate B", role: "Staff Designer", fit: 81 },
              { name: "Candidate C", role: "ML Engineer", fit: 74 },
            ].map((c, i) => (
              <motion.div
                key={c.name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 + i * 0.1 }}
                className="lift rounded-2xl border border-border bg-background p-4 text-left"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-semibold">{c.name}</div>
                    <div className="text-xs text-muted-foreground">{c.role}</div>
                  </div>
                  <div className="font-mono text-2xl font-semibold tabular-nums text-accent">
                    <AnimatedNumber value={c.fit} duration={1200} />
                  </div>
                </div>
                <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${c.fit}%` }}
                    transition={{ duration: 1.1, delay: 0.7 + i * 0.1 }}
                    className="h-full bg-violet-grad"
                  />
                </div>
                <div className="mt-3 flex flex-wrap gap-1">
                  {["React", "TS", "GraphQL"].map((s) => (
                    <span
                      key={s}
                      className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground"
                    >
                      {s}
                    </span>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </SpotlightCard>
      </motion.div>
    </section>
  );
}

// Capability strip — what HireFlow does, not fake "events". No names, no times,
// nothing that pretends to be a live activity feed.
const CAPABILITIES: { kind: string; text: string }[] = [
  { kind: "jd",        text: "AI Job Wizard drafts your JD" },
  { kind: "match",     text: "Explainable fit score per candidate" },
  { kind: "interview", text: "Async AI interview room" },
  { kind: "shortlist", text: "Bias check on every shortlist" },
  { kind: "match",     text: "Skill-graph matching, not keyword search" },
  { kind: "jd",        text: "Streaming JD generation" },
  { kind: "interview", text: "Live behavioral signals during interviews" },
  { kind: "hire",      text: "Calibrated, decision-ready reports" },
];

const TICKER_ICON: Record<string, typeof Sparkles> = {
  match: Sparkles,
  interview: Bot,
  hire: CheckCircle2,
  shortlist: Shield,
  jd: Zap,
};

function ActivityTicker() {
  const prefersReducedMotion =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  // Duplicate the list once and translate -50% so the loop is seamless.
  const items = [...CAPABILITIES, ...CAPABILITIES];
  return (
    <div
      className="relative mx-auto mt-10 max-w-3xl overflow-hidden rounded-full border border-border bg-card/40 py-2 backdrop-blur"
      role="list"
      aria-label="Platform capabilities"
    >
      <div
        className="pointer-events-none absolute inset-y-0 left-0 z-10 w-16 bg-gradient-to-r from-background to-transparent"
        aria-hidden
      />
      <div
        className="pointer-events-none absolute inset-y-0 right-0 z-10 w-16 bg-gradient-to-l from-background to-transparent"
        aria-hidden
      />
      <motion.div
        animate={prefersReducedMotion ? undefined : { x: ["0%", "-50%"] }}
        transition={
          prefersReducedMotion
            ? undefined
            : { duration: 38, ease: "linear", repeat: Infinity }
        }
        className="flex w-max gap-8 whitespace-nowrap text-xs text-muted-foreground"
      >
        {items.map((a, i) => {
          const Icon = TICKER_ICON[a.kind] ?? Sparkles;
          return (
            <span key={i} role="listitem" className="inline-flex items-center gap-2">
              <Icon className="h-3 w-3 text-accent" aria-hidden />
              <span>{a.text}</span>
              <span className="text-muted-foreground/40" aria-hidden>·</span>
            </span>
          );
        })}
      </motion.div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Pillars — three product claims. No fabricated metrics; if/when we collect
// real platform-wide aggregates we can promote them here.
// ─────────────────────────────────────────────────────────────────────────────

const PILLARS = [
  {
    icon: Zap,
    title: "Drafted, not boilerplated",
    body: "Job descriptions stream out tailored to your role, level, and team — not a template fill-in.",
  },
  {
    icon: Sparkles,
    title: "Explainable fit",
    body: "Every score breaks down into skills, trajectory, and culture, with the CV lines that earned each point.",
  },
  {
    icon: Shield,
    title: "Fair by design",
    body: "Bias watchdog audits every shortlist. Calibration tools keep scoring consistent across recruiters.",
  },
];

function StatsBand() {
  return (
    <motion.section
      viewport={{ once: true, margin: "-100px" }}
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="mx-auto max-w-6xl px-6 py-16"
    >
      <div className="grid gap-8 rounded-3xl border border-border bg-card/60 p-8 backdrop-blur md:grid-cols-3 md:p-12">
        {PILLARS.map((p, i) => (
          <motion.div
            key={p.title}
            initial={{ opacity: 0, y: 14 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: i * 0.1 }}
            className="text-center md:text-left"
          >
            <div className="mb-4 inline-grid h-12 w-12 place-items-center rounded-2xl bg-violet-grad text-white">
              <p.icon className="h-5 w-5" aria-hidden />
            </div>
            <h3 className="font-display text-2xl tracking-tight">{p.title}</h3>
            <p className="mt-2 max-w-[34ch] text-sm text-muted-foreground">
              {p.body}
            </p>
          </motion.div>
        ))}
      </div>
    </motion.section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Bento feature grid — 6 cards with mismatched spans for visual rhythm
// ─────────────────────────────────────────────────────────────────────────────

const FEATURES = [
  {
    icon: Sparkles,
    title: "AI Job Wizard",
    body:
      "Type a few notes. Get a polished, on-brand JD with calibrated must-haves and nice-to-haves — streaming as you watch.",
    span: "md:col-span-2",
    accent: true,
  },
  {
    icon: Users,
    title: "Explainable Ranking",
    body:
      "Fit scores broken down into skills, trajectory, and culture — with the exact lines from the CV that earned each point.",
    span: "md:col-span-1",
    accent: false,
  },
  {
    icon: Bot,
    title: "AI Interview Room",
    body:
      "Async structured interviews with a calm AI host. Behavioral signals + transcript + score, in one quiet room.",
    span: "md:col-span-1",
    accent: false,
  },
  {
    icon: BarChart3,
    title: "Pipeline Analytics",
    body:
      "Funnel velocity, source quality, time-to-decision. The metrics your hiring committee actually asks for.",
    span: "md:col-span-1",
    accent: false,
  },
  {
    icon: Zap,
    title: "Decision Queue",
    body:
      "Tinder-fast triage of new applicants with keyboard shortcuts. Never wonder who's waiting on you.",
    span: "md:col-span-1",
    accent: false,
  },
  {
    icon: Shield,
    title: "Bias Watchdog",
    body:
      "Per-shortlist parity checks across gender, geography, and seniority — with one-click rebalance suggestions.",
    span: "md:col-span-3",
    accent: true,
  },
];

function FeaturesBento() {
  return (
    <section id="product" className="mx-auto max-w-6xl px-6 py-24">
      <div className="mb-12 max-w-2xl">
        <div className="text-xs uppercase tracking-widest text-muted-foreground">
          The platform
        </div>
        <h2 className="font-display mt-2 text-4xl tracking-tight md:text-5xl">
          One workspace. Two sides. Zero friction.
        </h2>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {FEATURES.map((f, i) => (
          <motion.div
            key={f.title}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5, delay: i * 0.05 }}
            className={f.span}
          >
            <SpotlightCard
              className={`lift relative h-full overflow-hidden rounded-2xl border border-border p-6 ${
                f.accent ? "bg-card" : "bg-card/70"
              }`}
            >
              {f.accent && (
                <div
                  aria-hidden
                  className="absolute -right-12 -top-12 h-40 w-40 rounded-full bg-violet-grad opacity-10 blur-2xl"
                />
              )}
              <div className="mb-4 inline-grid h-10 w-10 place-items-center rounded-xl bg-violet-grad text-white">
                <f.icon className="h-5 w-5" />
              </div>
              <h3 className="text-lg font-semibold">{f.title}</h3>
              <p className="mt-2 max-w-prose text-sm text-muted-foreground">
                {f.body}
              </p>
            </SpotlightCard>
          </motion.div>
        ))}
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Workflow — left rail with a vertical accent line; right interview mock
// ─────────────────────────────────────────────────────────────────────────────

const STEPS = [
  {
    title: "Open the role",
    body: "Streaming AI JD wizard turns notes into a calibrated job description in 30 seconds.",
  },
  {
    title: "Let the room fill",
    body: "Candidates apply or get sourced. Fit scores appear with full evidence trails.",
  },
  {
    title: "Run structured interviews",
    body: "Async AI room asks the same questions, captures the same signals, scores fairly.",
  },
  {
    title: "Decide together",
    body: "Compare top 3 side-by-side, see bias checks, send the offer with one click.",
  },
];

function Workflow() {
  return (
    <section id="workflow" className="mx-auto max-w-6xl px-6 py-24">
      <div className="grid gap-12 md:grid-cols-2 md:items-start">
        <div>
          <div className="text-xs uppercase tracking-widest text-muted-foreground">
            Workflow
          </div>
          <h2 className="font-display mt-2 text-4xl tracking-tight md:text-5xl">
            From first signal to signed offer.
          </h2>

          <ol className="relative mt-10 space-y-7">
            {/* Vertical rail behind the numbered circles. */}
            <div
              aria-hidden
              className="absolute left-4 top-2 bottom-2 w-px bg-gradient-to-b from-accent/50 via-border to-transparent"
            />
            {STEPS.map((s, i) => (
              <motion.li
                key={s.title}
                initial={{ opacity: 0, x: -8 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: "-80px" }}
                transition={{ duration: 0.4, delay: i * 0.08 }}
                className="relative flex gap-4 pl-1"
              >
                <div className="relative z-10 grid h-8 w-8 shrink-0 place-items-center rounded-full border border-border bg-card font-mono text-xs shadow-card-soft">
                  {i + 1}
                </div>
                <div>
                  <div className="font-semibold">{s.title}</div>
                  <div className="mt-0.5 text-sm text-muted-foreground">{s.body}</div>
                </div>
              </motion.li>
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
            <div className="text-xs uppercase tracking-widest text-white/60">
              AI Interview Room
            </div>
            <div className="font-display mt-2 text-3xl">
              "Walk me through a system you designed end-to-end."
            </div>
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
              ].map((m, i) => (
                <motion.div
                  key={m.l}
                  initial={{ opacity: 0, y: 8 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: i * 0.08 }}
                  className="rounded-xl bg-white/10 p-3 backdrop-blur"
                >
                  <div className="text-white/70">{m.l}</div>
                  <div className="font-mono text-lg tabular-nums">
                    <AnimatedNumber value={m.v} duration={1100} />
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Testimonial — single anchor quote
// ─────────────────────────────────────────────────────────────────────────────

function Testimonial() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-20">
      <motion.figure
        initial={{ opacity: 0, y: 16 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-80px" }}
        transition={{ duration: 0.6 }}
        className="relative mx-auto max-w-3xl overflow-hidden rounded-3xl border border-border bg-card p-10 text-center md:p-14"
      >
        <div
          aria-hidden
          className="absolute -top-20 left-1/2 h-48 w-48 -translate-x-1/2 rounded-full bg-violet-grad opacity-20 blur-3xl"
        />
        <Quote className="mx-auto h-8 w-8 text-accent" />
        <blockquote className="font-display mt-4 text-pretty text-2xl leading-snug tracking-tight md:text-3xl">
          "We went from gut-feel triage to a calm, evidence-trailed pipeline.
          The first time the AI Interview Room caught a strong candidate I
          would have skipped, I knew this was different."
        </blockquote>
        <figcaption className="mt-6 inline-flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-full bg-violet-grad text-xs font-semibold text-accent-foreground">
            NB
          </div>
          <div className="text-left text-sm">
            <div className="font-semibold">Naomi Brooks</div>
            <div className="text-muted-foreground">
              Head of Talent · Lumen Labs
            </div>
          </div>
        </figcaption>
      </motion.figure>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Final CTA — large, glowing, with trust block as the right column
// ─────────────────────────────────────────────────────────────────────────────

function FinalCTA() {
  // Use scroll-progress to nudge the headline as it enters the viewport.
  const sectionRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ["start end", "end start"],
  });
  const headlineY = useTransform(scrollYProgress, [0, 1], [40, -40]);

  return (
    <section ref={sectionRef} id="trust" className="mx-auto max-w-6xl px-6 py-24">
      <SpotlightCard className="relative overflow-hidden rounded-3xl border border-border bg-card p-10 md:p-14">
        <div
          aria-hidden
          className="absolute -inset-1 rounded-3xl bg-violet-grad opacity-10 blur-3xl"
        />
        <div className="relative grid gap-10 md:grid-cols-2 md:items-center">
          <div>
            <div className="text-xs uppercase tracking-widest text-muted-foreground">
              Built for trust
            </div>
            <motion.h2
              style={{ y: headlineY }}
              className="font-display mt-2 text-3xl tracking-tight md:text-5xl"
            >
              Ready to make hiring feel <em className="italic text-accent">inevitable</em>?
            </motion.h2>
            <p className="mt-4 max-w-md text-pretty text-muted-foreground">
              Open the recruiter app and run a real shortlist in under five
              minutes. No credit card. Bring your own JD or let the wizard
              draft one.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <Link
                to="/app"
                className="press-tight glow-pulse group inline-flex items-center gap-1.5 rounded-full bg-violet-grad px-5 py-3 text-sm font-semibold text-accent-foreground shadow-glow transition-transform hover:scale-[1.04]"
              >
                Open recruiter app
                <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
              </Link>
              <Link
                to="/c"
                className="press-tight rounded-full border border-border bg-card/60 px-5 py-3 text-sm font-semibold backdrop-blur hover:bg-card"
              >
                Try the candidate side
              </Link>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            {[
              { k: "GDPR", v: "Compliant by design" },
              { k: "SOC 2", v: "Type II in progress" },
              { k: "Bias audit", v: "Per-shortlist, per-job" },
              { k: "Data", v: "Your tenant. Your keys." },
            ].map((t) => (
              <motion.div
                key={t.k}
                initial={{ opacity: 0, y: 8 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4 }}
                className="lift rounded-2xl border border-border bg-background p-4"
              >
                <div className="text-xs text-muted-foreground">{t.k}</div>
                <div className="mt-1 font-medium">{t.v}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </SpotlightCard>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Footer
// ─────────────────────────────────────────────────────────────────────────────

function Footer() {
  return (
    <footer className="border-t border-border">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-6 py-8 text-sm text-muted-foreground md:flex-row">
        <div>© 2026 HireFlow. Hiring with intention.</div>
        <div className="flex gap-5">
          <Link to="/app" className="hover:text-foreground">Recruiter app</Link>
          <Link to="/c" className="hover:text-foreground">Candidate app</Link>
        </div>
      </div>
    </footer>
  );
}

