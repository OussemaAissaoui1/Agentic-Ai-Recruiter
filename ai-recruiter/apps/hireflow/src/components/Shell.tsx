import { Link, Outlet, useLocation, useNavigate } from "@tanstack/react-router";
import { AnimatePresence, motion } from "framer-motion";
import {
  LayoutDashboard,
  Briefcase,
  Users,
  BarChart3,
  Sparkles,
  Search,
  Settings,
  Sun,
  Moon,
  Bell,
  Network,
} from "lucide-react";
import { useApp } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { CommandPalette } from "@/components/CommandPalette";
import { CopilotPanel } from "@/components/CopilotPanel";
import { RoleSwitcher } from "@/components/RoleSwitcher";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useMarkNotificationRead, useNotifications } from "@/lib/queries";
import { useEffect } from "react";
import { formatDistanceToNow } from "date-fns";

const HR_NAV = [
  { to: "/app", label: "Dashboard", icon: LayoutDashboard, exact: true },
  { to: "/app/jobs", label: "Jobs", icon: Briefcase },
  { to: "/app/applicants", label: "Applicants", icon: Users },
  { to: "/app/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/app/radar", label: "Talent Radar", icon: Sparkles },
  { to: "/app/explorer", label: "Graph Explorer", icon: Network },
  { to: "/app/settings", label: "Settings", icon: Settings },
];

export function HRShell() {
  const { theme, toggleTheme, setPaletteOpen, setCopilotOpen, role } = useApp();
  const loc = useLocation();
  const nav = useNavigate();

  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.classList.toggle("dark", theme === "dark");
    }
  }, [theme]);

  useEffect(() => {
    if (role === "candidate") nav({ to: "/c" });
  }, [role, nav]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen(true);
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "j") {
        e.preventDefault();
        setCopilotOpen(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setPaletteOpen, setCopilotOpen]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="fixed inset-0 -z-10 bg-aurora opacity-60" aria-hidden />
      <div className="flex">
        {/* Sidebar */}
        <aside className="sticky top-0 hidden h-screen w-60 shrink-0 border-r border-border bg-sidebar/60 backdrop-blur-xl md:flex md:flex-col">
          <div className="flex h-16 items-center gap-2 px-5">
            <Logo />
          </div>
          <nav className="flex-1 space-y-1 px-3 py-4">
            {HR_NAV.map((item) => {
              const active = item.exact
                ? loc.pathname === item.to
                : loc.pathname.startsWith(item.to);
              return (
                <Link
                  key={item.to}
                  to={item.to}
                  className={`group relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all ${
                    active
                      ? "bg-sidebar-accent text-sidebar-accent-foreground"
                      : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground"
                  }`}
                >
                  {active && (
                    <motion.div
                      layoutId="hr-nav-indicator"
                      className="absolute left-0 h-5 w-0.5 rounded-r bg-accent"
                    />
                  )}
                  <item.icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
          </nav>
          <div className="border-t border-border p-3">
            <RoleSwitcher />
          </div>
        </aside>

        {/* Main */}
        <div className="min-w-0 flex-1">
          <header className="sticky top-0 z-30 flex h-16 items-center gap-3 border-b border-border bg-background/70 px-6 backdrop-blur-xl">
            <button
              onClick={() => setPaletteOpen(true)}
              aria-label="Open command palette (Cmd+K)"
              className="group flex min-h-[40px] w-full max-w-md items-center gap-2 rounded-lg border border-border bg-muted/50 px-3 py-1.5 text-left text-sm text-muted-foreground transition hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              <Search className="h-4 w-4" aria-hidden />
              <span className="flex-1">Search candidates, jobs, anything…</span>
              <kbd className="hidden rounded bg-background px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground sm:inline">
                ⌘K
              </kbd>
            </button>
            <div className="ml-auto flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setCopilotOpen(true)}
                title="AI Copilot (⌘J)"
                aria-label="Open AI Copilot"
              >
                <Sparkles className="h-4 w-4" aria-hidden />
              </Button>
              <NotificationsBell role="hr" />
              <Button
                variant="ghost"
                size="icon"
                onClick={toggleTheme}
                title="Toggle theme"
                aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
              >
                {theme === "dark" ? (
                  <Sun className="h-4 w-4" aria-hidden />
                ) : (
                  <Moon className="h-4 w-4" aria-hidden />
                )}
              </Button>
              <div
                className="ml-2 hidden h-8 w-8 items-center justify-center rounded-full bg-violet-grad text-xs font-semibold text-accent-foreground md:flex"
                aria-hidden
              >
                <Users className="h-4 w-4" />
              </div>
            </div>
          </header>
          <main className="mx-auto max-w-[1400px] px-6 py-8">
            <AnimatePresence mode="wait">
              <motion.div
                key={loc.pathname}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.22, ease: [0.2, 0.8, 0.2, 1] }}
              >
                <Outlet />
              </motion.div>
            </AnimatePresence>
          </main>
        </div>
      </div>
      <CommandPalette />
      <CopilotPanel />
    </div>
  );
}

export function CandidateShell() {
  const { theme, toggleTheme, role } = useApp();
  const nav = useNavigate();
  const cloc = useLocation();
  useEffect(() => {
    if (role === "hr") nav({ to: "/app" });
  }, [role, nav]);
  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.classList.toggle("dark", theme === "dark");
    }
  }, [theme]);
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="fixed inset-0 -z-10 bg-aurora opacity-50" aria-hidden />
      <header className="sticky top-0 z-30 border-b border-border bg-background/70 backdrop-blur-xl">
        <div className="mx-auto flex h-16 w-full max-w-7xl items-center gap-6 px-6">
          <Logo />
          <nav className="ml-4 hidden items-center gap-4 text-sm md:flex">
            <Link to="/c" className="text-foreground hover:text-accent" activeOptions={{ exact: true }}>
              Discover
            </Link>
            <Link to="/c/applications" className="text-muted-foreground hover:text-foreground">
              My Applications
            </Link>
            <Link to="/c/profile" className="text-muted-foreground hover:text-foreground">
              Profile
            </Link>
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <NotificationsBell role="candidate" />
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
              title="Toggle theme"
            >
              {theme === "dark" ? (
                <Sun className="h-4 w-4" aria-hidden />
              ) : (
                <Moon className="h-4 w-4" aria-hidden />
              )}
            </Button>
            <RoleSwitcher compact />
          </div>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl px-6 py-10">
        <AnimatePresence mode="wait">
          <motion.div
            key={cloc.pathname}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.22, ease: [0.2, 0.8, 0.2, 1] }}
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

function Logo() {
  return (
    <Link to="/" className="flex items-center gap-2" aria-label="HireFlow home">
      <img
        src="/logo-mark.png"
        alt=""
        width={28}
        height={28}
        className="h-7 w-7 select-none object-contain"
        draggable={false}
      />
      <span className="font-display text-lg tracking-tight">HireFlow</span>
    </Link>
  );
}

function NotificationsBell({ role }: { role: "hr" | "candidate" }) {
  const { data: list = [] } = useNotifications({ user_role: role });
  const markRead = useMarkNotificationRead();
  const navigate = useNavigate();
  const unread = list.filter((n) => !n.read).length;

  const onClick = (id: string, read: boolean, link: string | null | undefined) => {
    if (!read) markRead.mutate(id);
    if (link) {
      void navigate({ to: link });
    }
  };

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          title="Notifications"
          aria-label={
            unread > 0
              ? `Notifications (${unread} unread)`
              : "Notifications"
          }
          className="relative"
        >
          <Bell className="h-4 w-4" aria-hidden />
          {unread > 0 && (
            <span
              className="absolute right-1 top-1 grid h-4 min-w-4 place-items-center rounded-full bg-accent px-1 text-[9px] font-semibold text-accent-foreground"
              aria-hidden
            >
              {unread > 9 ? "9+" : unread}
            </span>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 p-0">
        <div className="border-b border-border px-4 py-3 text-sm font-semibold">
          Notifications
        </div>
        <div className="max-h-80 overflow-y-auto">
          {list.length === 0 && (
            <div className="px-4 py-8 text-center text-xs text-muted-foreground">
              You're all caught up.
            </div>
          )}
          {list.map((n) => (
            <button
              key={n.id}
              onClick={() => onClick(n.id, n.read, n.link)}
              className={`block w-full border-b border-border px-4 py-3 text-left transition hover:bg-muted/40 ${
                n.read ? "opacity-70" : ""
              } ${n.link ? "cursor-pointer" : ""}`}
            >
              <div className="flex items-start gap-2">
                {!n.read && (
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
                )}
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium">{n.title}</div>
                  <div className="text-xs text-muted-foreground">{n.body}</div>
                  <div className="mt-1 text-[10px] uppercase tracking-widest text-muted-foreground">
                    {(() => {
                      try {
                        return formatDistanceToNow(new Date(n.created_at), {
                          addSuffix: true,
                        });
                      } catch {
                        return "";
                      }
                    })()}
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}
