import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Role = "hr" | "candidate";

export type CandidateProfile = {
  candidateName: string;
  candidateEmail: string;
  about: string;
  skills: string;
  years: string;
  lookingFor: string;
  comp: string;
  cvFilename: string | null;
  cvText: string | null;
};

interface AppState {
  role: Role;
  setRole: (r: Role) => void;
  theme: "light" | "dark";
  toggleTheme: () => void;
  paletteOpen: boolean;
  setPaletteOpen: (v: boolean) => void;
  copilotOpen: boolean;
  setCopilotOpen: (v: boolean) => void;

  // Candidate identity & profile
  candidateName: string;
  candidateEmail: string;
  setCandidate: (v: { name?: string; email?: string }) => void;

  profile: CandidateProfile;
  setProfile: (patch: Partial<CandidateProfile>) => void;
}

const defaultProfile: CandidateProfile = {
  candidateName: "",
  candidateEmail: "",
  about:
    "Senior frontend engineer with 7 years shipping design-led product surfaces. Past: Linear-style platforms, design systems, motion craft.",
  skills: "React · TypeScript · Tailwind · Framer Motion · GraphQL",
  years: "7",
  lookingFor: "Senior IC, EU remote",
  comp: "$140–180k",
  cvFilename: null,
  cvText: null,
};

export const useApp = create<AppState>()(
  persist(
    (set) => ({
      role: "hr",
      setRole: (role) => set({ role }),
      theme: "light",
      toggleTheme: () =>
        set((s) => {
          const next = s.theme === "light" ? "dark" : "light";
          if (typeof document !== "undefined") {
            document.documentElement.classList.toggle("dark", next === "dark");
          }
          return { theme: next };
        }),
      paletteOpen: false,
      setPaletteOpen: (paletteOpen) => set({ paletteOpen }),
      copilotOpen: false,
      setCopilotOpen: (copilotOpen) => set({ copilotOpen }),

      candidateName: "",
      candidateEmail: "",
      setCandidate: ({ name, email }) =>
        set((s) => ({
          candidateName: name ?? s.candidateName,
          candidateEmail: email ?? s.candidateEmail,
          profile: {
            ...s.profile,
            candidateName: name ?? s.profile.candidateName,
            candidateEmail: email ?? s.profile.candidateEmail,
          },
        })),

      profile: defaultProfile,
      setProfile: (patch) =>
        set((s) => ({
          profile: { ...s.profile, ...patch },
          candidateName: patch.candidateName ?? s.candidateName,
          candidateEmail: patch.candidateEmail ?? s.candidateEmail,
        })),
    }),
    { name: "hireflow-app" },
  ),
);
