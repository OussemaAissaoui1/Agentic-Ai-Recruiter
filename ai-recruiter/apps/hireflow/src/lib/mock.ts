// Deterministic mock data for HireFlow.

export type Job = {
  id: string;
  title: string;
  team: string;
  location: string;
  type: "Full-time" | "Contract" | "Part-time";
  status: "Open" | "Draft" | "Closed";
  applicants: number;
  newToday: number;
  posted: string;
  salaryMin: number;
  salaryMax: number;
  description: string;
  mustHaves: string[];
  niceToHaves: string[];
};

export type Candidate = {
  id: string;
  name: string;
  initials: string;
  title: string;
  location: string;
  fit: number; // 0-100
  jobId: string;
  stage: "New" | "Screen" | "Interview" | "Offer" | "Rejected";
  experience: number;
  skills: string[];
  matchedSkills: string[];
  missingSkills: string[];
  reasoning: string;
  highlights: string[];
  salary: number;
  available: string;
  avatarHue: number;
  scores: { skills: number; experience: number; culture: number; trajectory: number };
};

const FIRST = ["Aria","Ben","Cleo","Diego","Elena","Felix","Gia","Hugo","Iris","Jamal","Kira","Leo","Maya","Nico","Otto","Priya","Quinn","Rhea","Sam","Tara","Uma","Vik","Wren","Yuki","Zane"];
const LAST = ["Park","Okafor","Müller","Costa","Reyes","Kim","Patel","Nguyen","Silva","Levi","Tan","Mori","Cohen","Singh","Larsen","Adeyemi","Rossi","Suzuki","Khan","Dubois"];
const TITLES = ["Senior Frontend Engineer","Staff Product Designer","ML Research Engineer","Growth PM","Head of Brand","DevRel Lead","Backend Engineer","Data Scientist","UX Researcher","Engineering Manager"];
const TEAMS = ["Platform","Growth","Design","Research","Infra","Brand"];
const LOCS = ["Remote — EU","Remote — Global","Berlin, DE","London, UK","Lisbon, PT","NYC, US","SF, US","Singapore"];
const SKILLS = ["React","TypeScript","Node","GraphQL","Postgres","Rust","Go","Python","PyTorch","Figma","Design Systems","A/B Testing","SQL","Kubernetes","AWS","Tailwind","Next.js","Framer Motion","Mixpanel","Amplitude"];

function rand(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}
const pick = <T>(r: () => number, arr: T[]) => arr[Math.floor(r() * arr.length)];
const sample = <T>(r: () => number, arr: T[], n: number) => {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, n);
};

export const JOBS: Job[] = Array.from({ length: 8 }).map((_, i) => {
  const r = rand(101 + i);
  const title = TITLES[i % TITLES.length];
  const must = sample(r, SKILLS, 4);
  const nice = sample(r, SKILLS.filter((s) => !must.includes(s)), 4);
  return {
    id: `job-${i + 1}`,
    title,
    team: pick(r, TEAMS),
    location: pick(r, LOCS),
    type: "Full-time",
    status: i === 7 ? "Draft" : "Open",
    applicants: 24 + Math.floor(r() * 180),
    newToday: Math.floor(r() * 14),
    posted: `${1 + Math.floor(r() * 21)}d ago`,
    salaryMin: 90 + Math.floor(r() * 40),
    salaryMax: 140 + Math.floor(r() * 80),
    description: `We're looking for a ${title.toLowerCase()} to join our ${pick(r, TEAMS)} team. You'll partner with cross-functional leaders to ship high-impact work that shapes the product.`,
    mustHaves: must,
    niceToHaves: nice,
  };
});

export const CANDIDATES: Candidate[] = Array.from({ length: 64 }).map((_, i) => {
  const r = rand(7 + i * 13);
  const first = pick(r, FIRST);
  const last = pick(r, LAST);
  const job = JOBS[i % JOBS.length];
  const matched = sample(r, job.mustHaves, 2 + Math.floor(r() * 3));
  const missing = job.mustHaves.filter((s) => !matched.includes(s));
  const extra = sample(r, SKILLS.filter((s) => !job.mustHaves.includes(s)), 3);
  const skills = [...matched, ...extra];
  const fit = Math.round(45 + r() * 50);
  const scores = {
    skills: Math.round(40 + r() * 55),
    experience: Math.round(40 + r() * 55),
    culture: Math.round(50 + r() * 45),
    trajectory: Math.round(40 + r() * 55),
  };
  return {
    id: `cand-${i + 1}`,
    name: `${first} ${last}`,
    initials: `${first[0]}${last[0]}`,
    title: pick(r, TITLES),
    location: pick(r, LOCS),
    fit,
    jobId: job.id,
    stage: pick(r, ["New","New","Screen","Interview","Offer","Rejected"] as Candidate["stage"][]),
    experience: 2 + Math.floor(r() * 12),
    skills,
    matchedSkills: matched,
    missingSkills: missing,
    reasoning: `${first} matches on ${matched.slice(0,2).join(" and ")}. Experience trajectory shows ${Math.floor(r()*3)+1} promotions in the last 4 years. ${missing.length ? `Gap: ${missing[0]}.` : "No critical skill gaps."}`,
    highlights: [
      `Shipped ${pick(r,["onboarding","checkout","search","analytics"])} that lifted ${pick(r,["activation","retention","conversion"])} +${5+Math.floor(r()*22)}%`,
      `Led team of ${2+Math.floor(r()*8)} across ${pick(r,["2","3","4"])} timezones`,
      `Open-source: ${300+Math.floor(r()*4000)} GitHub stars`,
    ],
    salary: 80 + Math.floor(r() * 90),
    available: pick(r, ["Immediately","2 weeks","1 month","2 months"]),
    avatarHue: Math.floor(r() * 360),
    scores,
  };
});

export const ANALYTICS = {
  funnel: [
    { stage: "Sourced", value: 1240 },
    { stage: "Applied", value: 612 },
    { stage: "Screened", value: 248 },
    { stage: "Interviewed", value: 96 },
    { stage: "Offered", value: 28 },
    { stage: "Hired", value: 19 },
  ],
  trend: Array.from({ length: 14 }).map((_, i) => ({
    day: `D${i + 1}`,
    applicants: 30 + Math.round(Math.sin(i / 2) * 12 + i * 1.6),
    quality: 50 + Math.round(Math.cos(i / 3) * 10 + i * 1.2),
  })),
  sources: [
    { name: "Inbound", value: 38 },
    { name: "Referral", value: 24 },
    { name: "LinkedIn", value: 22 },
    { name: "Sourced", value: 16 },
  ],
  pipeline: [
    { team: "Platform", open: 12, hired: 4 },
    { team: "Growth", open: 8, hired: 3 },
    { team: "Design", open: 5, hired: 2 },
    { team: "Research", open: 4, hired: 1 },
    { team: "Infra", open: 7, hired: 5 },
  ],
};

export const INTERVIEW_QUESTIONS = [
  "Tell me about a time you shipped something against the odds.",
  "Walk me through a system you designed end-to-end.",
  "How do you decide what NOT to build?",
  "Describe a disagreement with a senior leader and how it resolved.",
  "What's a technical opinion you've changed in the last year?",
];
