import { useApp } from "@/lib/store";
import { Command } from "cmdk";
import { Briefcase, Users, BarChart3, Sparkles, FilePlus, MessageCircle, Sun, Moon } from "lucide-react";
import { useNavigate } from "@tanstack/react-router";
import { JOBS, CANDIDATES } from "@/lib/mock";

export function CommandPalette() {
  const { paletteOpen, setPaletteOpen, toggleTheme, setCopilotOpen } = useApp();
  const nav = useNavigate();

  if (!paletteOpen) return null;

  const go = (path: string) => {
    setPaletteOpen(false);
    nav({ to: path });
  };

  return (
    <div
      className="fixed inset-0 z-50 grid place-items-start bg-foreground/20 px-4 pt-[12vh] backdrop-blur-sm"
      onClick={() => setPaletteOpen(false)}
    >
      <div
        className="w-full max-w-xl overflow-hidden rounded-2xl border border-border bg-popover shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <Command>
          <Command.Input
            autoFocus
            placeholder="Type a command, jump to a candidate, or ask anything…"
            className="w-full border-b border-border bg-transparent px-4 py-3.5 text-sm outline-none placeholder:text-muted-foreground"
          />
          <Command.List className="max-h-[60vh] overflow-y-auto p-2">
            <Command.Empty className="px-4 py-6 text-center text-sm text-muted-foreground">
              No results.
            </Command.Empty>
            <Command.Group heading="Actions" className="px-2 py-1 text-xs text-muted-foreground">
              <PItem onSelect={() => go("/app/jobs/new")} icon={<FilePlus className="h-4 w-4" />}>
                Create new job with AI
              </PItem>
              <PItem onSelect={() => { setPaletteOpen(false); setCopilotOpen(true); }} icon={<Sparkles className="h-4 w-4" />}>
                Open AI Copilot
              </PItem>
              <PItem onSelect={() => { toggleTheme(); setPaletteOpen(false); }} icon={<Sun className="h-4 w-4" />}>
                Toggle theme
              </PItem>
            </Command.Group>
            <Command.Group heading="Navigate" className="px-2 py-1 text-xs text-muted-foreground">
              <PItem onSelect={() => go("/app")} icon={<BarChart3 className="h-4 w-4" />}>Dashboard</PItem>
              <PItem onSelect={() => go("/app/jobs")} icon={<Briefcase className="h-4 w-4" />}>Jobs</PItem>
              <PItem onSelect={() => go("/app/applicants")} icon={<Users className="h-4 w-4" />}>Applicants</PItem>
              <PItem onSelect={() => go("/app/radar")} icon={<Sparkles className="h-4 w-4" />}>Talent Radar</PItem>
            </Command.Group>
            <Command.Group heading="Jobs" className="px-2 py-1 text-xs text-muted-foreground">
              {JOBS.slice(0, 4).map((j) => (
                <PItem key={j.id} onSelect={() => go(`/app/jobs/${j.id}`)} icon={<Briefcase className="h-4 w-4" />}>
                  {j.title} · <span className="text-muted-foreground">{j.team}</span>
                </PItem>
              ))}
            </Command.Group>
            <Command.Group heading="Candidates" className="px-2 py-1 text-xs text-muted-foreground">
              {CANDIDATES.slice(0, 5).map((c) => (
                <PItem key={c.id} onSelect={() => go(`/app/applicants?c=${c.id}`)} icon={<MessageCircle className="h-4 w-4" />}>
                  {c.name} · <span className="text-muted-foreground">{c.title}</span>
                </PItem>
              ))}
            </Command.Group>
          </Command.List>
        </Command>
      </div>
    </div>
  );
}

function PItem({ onSelect, icon, children }: { onSelect: () => void; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <Command.Item
      onSelect={onSelect}
      className="flex cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-sm text-foreground aria-selected:bg-accent/15 aria-selected:text-foreground"
    >
      <span className="text-muted-foreground">{icon}</span>
      <span className="flex-1 truncate">{children}</span>
    </Command.Item>
  );
}
