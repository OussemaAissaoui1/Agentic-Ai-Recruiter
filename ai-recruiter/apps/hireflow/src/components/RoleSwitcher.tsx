import { useApp } from "@/lib/store";
import { motion } from "framer-motion";
import { Briefcase, User2 } from "lucide-react";

export function RoleSwitcher({ compact = false }: { compact?: boolean }) {
  const { role, setRole } = useApp();
  return (
    <div
      className={`relative grid grid-cols-2 rounded-full border border-border bg-card/60 p-0.5 text-xs font-medium ${
        compact ? "w-[170px]" : "w-full"
      }`}
    >
      {(["hr", "candidate"] as const).map((r) => {
        const active = role === r;
        return (
          <button
            key={r}
            onClick={() => setRole(r)}
            className={`relative z-10 flex items-center justify-center gap-1.5 rounded-full px-3 py-1.5 transition-colors ${
              active ? "text-accent-foreground" : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {active && (
              <motion.div
                layoutId="role-pill"
                className="absolute inset-0 -z-10 rounded-full bg-violet-grad"
                transition={{ type: "spring", stiffness: 400, damping: 30 }}
              />
            )}
            {r === "hr" ? <Briefcase className="h-3 w-3" /> : <User2 className="h-3 w-3" />}
            {r === "hr" ? "Recruiter" : "Candidate"}
          </button>
        );
      })}
    </div>
  );
}
