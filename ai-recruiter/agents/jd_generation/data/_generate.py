"""Deterministic generator for the shipped synthetic dataset.

Produces:
    synthetic_employees.json   — 100 employees + supporting graph entities
    esco_skills_subset.json    — ~500-entry skill catalog

Run from repo root:
    python -m agents.jd_generation.data._generate

The generator is seeded so every run produces identical output. Re-run only
when you intentionally want to change the shipped data.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

OUT_DIR = Path(__file__).resolve().parent
SEED = 1337

random.seed(SEED)

# ---------------------------------------------------------------------------
# Role families and per-family role configuration
# ---------------------------------------------------------------------------
ROLE_FAMILIES = ["backend", "frontend", "data_science", "product", "design"]
LEVELS = ["junior", "mid", "senior"]

ROLE_TITLES = {
    "backend":      {"junior": "Junior Backend Engineer",
                     "mid":    "Backend Engineer",
                     "senior": "Senior Backend Engineer"},
    "frontend":     {"junior": "Junior Frontend Engineer",
                     "mid":    "Frontend Engineer",
                     "senior": "Senior Frontend Engineer"},
    "data_science": {"junior": "Junior Data Scientist",
                     "mid":    "Data Scientist",
                     "senior": "Senior Data Scientist"},
    "product":      {"junior": "Associate Product Manager",
                     "mid":    "Product Manager",
                     "senior": "Senior Product Manager"},
    "design":       {"junior": "Junior Product Designer",
                     "mid":    "Product Designer",
                     "senior": "Senior Product Designer"},
}

DEPARTMENT = {
    "backend": "engineering", "frontend": "engineering",
    "data_science": "data", "product": "product", "design": "design",
}

# ---------------------------------------------------------------------------
# Skill catalog — ~500 entries broken into role-relevant clusters
# ---------------------------------------------------------------------------
SKILL_CLUSTERS: Dict[str, List[str]] = {
    "language_python":     ["Python", "Type hints", "Pytest", "asyncio", "Flask", "FastAPI", "Django",
                            "SQLAlchemy", "Celery", "Pydantic", "Click", "Poetry", "ruff", "mypy"],
    "language_js":         ["JavaScript", "TypeScript", "Node.js", "Express", "Jest", "Vitest",
                            "Webpack", "Vite", "ESLint", "Prettier", "npm", "pnpm", "Yarn"],
    "language_systems":    ["Go", "Rust", "C", "C++", "Cargo", "Goroutines", "Tokio", "gRPC",
                            "Protocol Buffers"],
    "language_jvm":        ["Java", "Kotlin", "Scala", "Maven", "Gradle", "Spring Boot", "JUnit"],
    "language_other":      ["Ruby", "Rails", "PHP", "Laravel", "Elixir", "Phoenix", "Swift",
                            "Objective-C"],
    "frontend_framework":  ["React", "Vue", "Svelte", "Angular", "Next.js", "Nuxt", "Remix",
                            "SvelteKit", "Astro", "Gatsby", "Solid.js"],
    "frontend_styling":    ["CSS", "Sass", "Less", "Tailwind CSS", "Styled Components", "Emotion",
                            "PostCSS", "BEM", "CSS Grid", "Flexbox", "Responsive design",
                            "Mobile-first design"],
    "frontend_a11y":       ["WAI-ARIA", "Accessibility audits", "Screen reader testing",
                            "Keyboard navigation", "WCAG 2.1", "Color contrast", "Focus management"],
    "frontend_state":      ["Redux", "Zustand", "Jotai", "Recoil", "MobX", "React Query",
                            "SWR", "TanStack Query", "TanStack Router"],
    "frontend_perf":       ["Code splitting", "Tree shaking", "Lazy loading", "Web Vitals",
                            "Lighthouse", "Bundle analysis", "Critical CSS", "Service Workers"],
    "frontend_other":      ["HTML5", "DOM APIs", "Browser APIs", "Cross-browser testing",
                            "Storybook", "Cypress", "Playwright", "Chromatic"],
    "backend_storage":     ["PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Memcached",
                            "Cassandra", "DynamoDB", "Elasticsearch", "ClickHouse", "Snowflake"],
    "backend_messaging":   ["Kafka", "RabbitMQ", "SQS", "Pub/Sub", "NATS", "Event-driven architecture"],
    "backend_apis":        ["REST", "GraphQL", "WebSockets", "Server-Sent Events", "OpenAPI",
                            "JSON Schema", "API versioning", "Rate limiting", "OAuth2", "JWT"],
    "backend_patterns":    ["Microservices", "Monolith decomposition", "Domain-driven design",
                            "CQRS", "Event sourcing", "Saga pattern", "Hexagonal architecture",
                            "Clean architecture"],
    "backend_perf":        ["Query optimization", "Database indexing", "Caching strategies",
                            "Load testing", "Profiling", "Distributed tracing"],
    "devops_cloud":        ["AWS", "GCP", "Azure", "Cloudflare", "DigitalOcean", "Fly.io",
                            "Vercel", "Netlify"],
    "devops_containers":   ["Docker", "Kubernetes", "Helm", "Docker Compose", "containerd",
                            "Buildpacks"],
    "devops_ci":           ["GitHub Actions", "GitLab CI", "CircleCI", "Jenkins", "ArgoCD",
                            "Terraform", "Pulumi", "Ansible"],
    "devops_observ":       ["Prometheus", "Grafana", "Datadog", "Sentry", "OpenTelemetry",
                            "ELK stack", "Loki", "Jaeger", "PagerDuty"],
    "data_ml_core":        ["NumPy", "pandas", "scikit-learn", "PyTorch", "TensorFlow", "Keras",
                            "JAX", "XGBoost", "LightGBM", "CatBoost"],
    "data_ml_advanced":    ["Hugging Face Transformers", "spaCy", "NLTK", "sentence-transformers",
                            "LangChain", "vLLM", "ONNX", "TensorRT", "Quantization", "Distillation"],
    "data_eng":            ["dbt", "Airflow", "Prefect", "Dagster", "Spark", "Flink",
                            "Beam", "Kafka Streams"],
    "data_stats":          ["A/B testing", "Statistical inference", "Bayesian methods",
                            "Causal inference", "Time series analysis", "Survival analysis",
                            "Experimentation design"],
    "data_viz":            ["matplotlib", "seaborn", "plotly", "D3.js", "Tableau", "Looker",
                            "Mode", "Hex", "Observable"],
    "data_warehouse":      ["BigQuery", "Snowflake", "Redshift", "Athena", "Presto", "Trino"],
    "product_strategy":    ["Product roadmap", "OKR setting", "Product vision", "Market analysis",
                            "Competitive analysis", "Pricing strategy", "Go-to-market"],
    "product_discovery":   ["User research", "Customer interviews", "Survey design",
                            "Jobs-to-be-done", "User personas", "Journey mapping",
                            "Usability testing", "Persona development"],
    "product_execution":   ["Agile", "Scrum", "Kanban", "Story mapping", "PRD writing",
                            "Backlog grooming", "Sprint planning", "Stakeholder management"],
    "product_analytics":   ["Mixpanel", "Amplitude", "Heap", "Pendo", "Google Analytics",
                            "Funnel analysis", "Cohort analysis", "Retention analysis"],
    "design_visual":       ["Figma", "Sketch", "Adobe XD", "Illustrator", "Photoshop",
                            "Typography", "Color theory", "Iconography", "Brand systems"],
    "design_interaction":  ["Wireframing", "Prototyping", "Information architecture",
                            "Interaction design", "Motion design", "Microinteractions"],
    "design_systems":      ["Design systems", "Component libraries", "Design tokens",
                            "Atomic design", "Style guides", "Pattern libraries"],
    "design_research":     ["User research methodologies", "Card sorting", "Tree testing",
                            "Eye tracking", "Heuristic evaluation", "Accessibility design"],
    "soft_skills":         ["Written communication", "Mentorship", "Technical leadership",
                            "Cross-functional collaboration", "Public speaking", "Async work",
                            "Code review", "Documentation"],
}

# Skills strongly associated with each role family — these will appear with
# high frequency in cohorts of that family.
FAMILY_CORE_CLUSTERS = {
    "backend":      ["language_python", "language_systems", "backend_storage", "backend_apis",
                     "backend_patterns", "backend_perf", "devops_containers", "devops_ci"],
    "frontend":     ["language_js", "frontend_framework", "frontend_styling", "frontend_state",
                     "frontend_perf", "frontend_a11y", "frontend_other"],
    "data_science": ["language_python", "data_ml_core", "data_ml_advanced", "data_eng",
                     "data_stats", "data_viz", "data_warehouse"],
    "product":      ["product_strategy", "product_discovery", "product_execution",
                     "product_analytics"],
    "design":       ["design_visual", "design_interaction", "design_systems", "design_research",
                     "frontend_a11y"],
}

ADJACENT_CLUSTERS = ["soft_skills", "devops_cloud", "devops_observ"]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_skill_catalog() -> List[Dict[str, Any]]:
    """Flatten clusters into a single ESCO-style catalog."""
    out: List[Dict[str, Any]] = []
    for cluster_name, items in SKILL_CLUSTERS.items():
        for i, name in enumerate(items):
            out.append({
                "id":       f"sk_{cluster_name}_{i:02d}",
                "name":     name,
                "category": cluster_name,
                "esco_id":  f"esco/{cluster_name}/{i:03d}",
            })
    return out


def build_roles() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for family in ROLE_FAMILIES:
        for level in LEVELS:
            out.append({
                "id":          f"role_{family}_{level}",
                "title":       ROLE_TITLES[family][level],
                "level":       level,
                "role_family": family,
                "department":  DEPARTMENT[family],
            })
    return out


def build_teams() -> List[Dict[str, Any]]:
    return [
        {"id": "team_platform",  "name": "Platform",   "department": "engineering", "size": 18},
        {"id": "team_growth",    "name": "Growth",     "department": "engineering", "size": 14},
        {"id": "team_insights",  "name": "Insights",   "department": "data",        "size": 10},
        {"id": "team_pm",        "name": "Product",    "department": "product",     "size": 8},
        {"id": "team_design",    "name": "Design",     "department": "design",      "size": 6},
    ]


def family_for_team(team_id: str) -> List[str]:
    """Which role families typically sit on this team."""
    return {
        "team_platform":  ["backend"],
        "team_growth":    ["frontend", "backend"],
        "team_insights":  ["data_science"],
        "team_pm":        ["product"],
        "team_design":    ["design"],
    }[team_id]


def sample_skills(rng: random.Random, family: str, level: str,
                  catalog_by_cluster: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Build a skills bundle for one employee.

    Universal-of-family skills appear in nearly every employee of that family;
    distinctive skills appear in some; long-tail skills appear rarely. Levels
    add proficiency mass to existing skills, not raw count.
    """
    bundle: List[Dict[str, Any]] = []

    core_clusters = FAMILY_CORE_CLUSTERS[family]
    # Universal: pick 4-6 of the core clusters and grab 2-3 skills each
    universal_clusters = rng.sample(core_clusters, k=min(5, len(core_clusters)))
    for cluster in universal_clusters:
        skills = catalog_by_cluster[cluster]
        for sk in rng.sample(skills, k=min(rng.randint(2, 3), len(skills))):
            base = rng.randint(3, 4)
            bump = {"junior": 0, "mid": 0, "senior": 1}[level]
            prof = min(5, base + bump)
            bundle.append({"skill_id": sk["id"], "proficiency": prof})

    # Distinctive: 2-3 skills from the other core clusters
    distinctive_clusters = [c for c in core_clusters if c not in universal_clusters]
    for cluster in rng.sample(distinctive_clusters, k=min(2, len(distinctive_clusters))):
        skills = catalog_by_cluster[cluster]
        if skills:
            sk = rng.choice(skills)
            bundle.append({"skill_id": sk["id"], "proficiency": rng.randint(2, 4)})

    # Long-tail: 1-2 skills from outside the family — adjacent clusters
    for cluster in rng.sample(ADJACENT_CLUSTERS, k=min(2, len(ADJACENT_CLUSTERS))):
        skills = catalog_by_cluster[cluster]
        if skills:
            sk = rng.choice(skills)
            bundle.append({"skill_id": sk["id"], "proficiency": rng.randint(2, 3)})

    # Dedup by skill_id — keep the max proficiency seen
    by_id: Dict[str, int] = {}
    for entry in bundle:
        sid = entry["skill_id"]
        by_id[sid] = max(by_id.get(sid, 0), entry["proficiency"])
    return [{"skill_id": sid, "proficiency": p} for sid, p in by_id.items()]


def pick_education(rng: random.Random) -> Dict[str, Any]:
    """70% CS, 20% mixed eng, 10% self-taught (no Education node)."""
    bucket = rng.random()
    if bucket < 0.70:
        return {
            "id":               f"edu_cs_{rng.randint(0, 99):02d}",
            "degree":           rng.choice(["BSc", "MSc", "PhD"]),
            "field":            "Computer Science",
            "institution_tier": rng.choice(["tier_1", "tier_2", "tier_2", "tier_3"]),
        }
    if bucket < 0.90:
        return {
            "id":               f"edu_eng_{rng.randint(0, 99):02d}",
            "degree":           rng.choice(["BSc", "MSc"]),
            "field":            rng.choice(["Electrical Engineering", "Mechanical Engineering",
                                            "Industrial Engineering", "Information Systems",
                                            "Mathematics", "Statistics", "Physics"]),
            "institution_tier": rng.choice(["tier_1", "tier_2", "tier_3"]),
        }
    return {}  # self-taught — no Education node


PRIOR_COMPANIES = [
    ("Stripe", "fintech", "large"), ("Shopify", "ecommerce", "large"),
    ("Notion", "saas", "mid"), ("Linear", "saas", "small"),
    ("Vercel", "devtools", "mid"), ("Datadog", "saas", "large"),
    ("Airbnb", "marketplace", "large"), ("DoorDash", "marketplace", "large"),
    ("OpenAI", "ai", "mid"), ("Anthropic", "ai", "small"),
    ("Hugging Face", "ai", "small"), ("Modal", "infra", "small"),
    ("Snowflake", "data", "large"), ("Databricks", "data", "large"),
    ("Local startup A", "saas", "small"), ("Local startup B", "fintech", "small"),
    ("Mid-cap fintech", "fintech", "mid"), ("Bigtech (FAANG)", "bigtech", "huge"),
    ("Bigtech adjacent", "bigtech", "huge"),
]


def pick_prior_companies(rng: random.Random, n: int) -> List[Dict[str, Any]]:
    sampled = rng.sample(PRIOR_COMPANIES, k=min(n, len(PRIOR_COMPANIES)))
    return [
        {"id":           f"co_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
         "name":         name,
         "industry":     industry,
         "size_bucket":  size}
        for (name, industry, size) in sampled
    ]


MIN_PER_LEVEL = 3   # ensure every (family, level) cohort meets the agent's MIN_COHORT


def _allocate_levels(rng: random.Random, total: int) -> List[str]:
    """Return a list of length `total` with one level per slot.

    Guarantees `MIN_PER_LEVEL` employees at each level (junior/mid/senior),
    then fills the rest by weighted random — biased toward `mid` to match
    realistic org pyramids. The result is shuffled so insertion order
    doesn't cluster all juniors at the start.
    """
    if total < MIN_PER_LEVEL * len(LEVELS):
        raise ValueError(
            f"can't stratify {total} employees across {len(LEVELS)} levels "
            f"with floor {MIN_PER_LEVEL}"
        )
    allocation: List[str] = []
    for lvl in LEVELS:
        allocation.extend([lvl] * MIN_PER_LEVEL)
    remaining = total - len(allocation)
    weights = [0.30, 0.45, 0.25]  # junior/mid/senior
    allocation.extend(rng.choices(LEVELS, weights=weights, k=remaining))
    rng.shuffle(allocation)
    return allocation


def build_employees(catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(SEED + 1)
    catalog_by_cluster: Dict[str, List[Dict[str, Any]]] = {}
    for sk in catalog:
        catalog_by_cluster.setdefault(sk["category"], []).append(sk)

    # Headcount per family — backend/frontend are largest, design smallest.
    # Each family is stratified so every level has at least MIN_PER_LEVEL members.
    counts = {"backend": 30, "frontend": 25, "data_science": 20, "product": 15, "design": 10}
    employees: List[Dict[str, Any]] = []

    today = datetime(2026, 5, 1)
    for family, n in counts.items():
        levels_for_family = _allocate_levels(rng, n)
        for level in levels_for_family:
            tenure_years = round(max(0.2, rng.gammavariate(2.0, 1.6)), 1)
            hire_date = today - timedelta(days=int(tenure_years * 365))

            # 30% have a prior internal role (PREVIOUSLY_HELD)
            prev_levels = []
            if rng.random() < 0.30 and level != "junior":
                prev_levels.append("junior" if level == "mid" else "mid")

            # Team assignment biased by family
            candidate_teams = [t for t in build_teams() if family in family_for_team(t["id"])]
            team = rng.choice(candidate_teams) if candidate_teams else None

            edu = pick_education(rng)
            education = [edu] if edu else []
            prior_n = rng.choices([0, 1, 2, 3], weights=[0.10, 0.30, 0.40, 0.20], k=1)[0]
            prior_companies = pick_prior_companies(rng, prior_n)

            emp_id = f"emp_{family}_{len(employees):03d}"
            employees.append({
                "id":              emp_id,
                "hire_date":       hire_date.date().isoformat(),
                "tenure_years":    tenure_years,
                "level":           level,
                "status":          "active",
                "roles_held":      [{
                    "role_id":     f"role_{family}_{level}",
                    "start":       hire_date.date().isoformat(),
                    "end":         None,
                    "current":     True,
                }] + [{
                    "role_id":     f"role_{family}_{prev}",
                    "start":       (hire_date - timedelta(days=int(rng.uniform(400, 800)))).date().isoformat(),
                    "end":         hire_date.date().isoformat(),
                    "current":     False,
                } for prev in prev_levels],
                "skills":          sample_skills(rng, family, level, catalog_by_cluster),
                "team_id":         team["id"] if team else None,
                "education":       education,
                "prior_companies": prior_companies,
            })
    return employees


def build_past_jds() -> List[Dict[str, Any]]:
    """10 past JDs: 7 good_hire, 3 rejected with realistic reason taxonomy."""
    today = datetime(2026, 5, 1)
    good = [
        {"id": "jd_past_be_01", "role_family": "backend",
         "text": ("We're hiring a senior backend engineer to own our event-ingestion "
                  "pipeline end-to-end. You'll partner with the data team on schema "
                  "evolution and bring rigor to our async story. Strong async Python, "
                  "Kafka or NATS experience, and pragmatic taste in failure modes."),
         "status": "approved", "hire_outcome": "good_hire"},
        {"id": "jd_past_be_02", "role_family": "backend",
         "text": ("Backend engineer for the platform team. You'll work across our "
                  "service mesh, our internal RPC layer, and our migration off the "
                  "monolith. Bias toward clarity over cleverness."),
         "status": "approved", "hire_outcome": "good_hire"},
        {"id": "jd_past_fe_01", "role_family": "frontend",
         "text": ("Senior frontend engineer for Growth. You'll own the funnel from "
                  "first paint to subscribed customer. We care about Web Vitals, "
                  "accessibility, and shipping fast without breaking things."),
         "status": "approved", "hire_outcome": "good_hire"},
        {"id": "jd_past_fe_02", "role_family": "frontend",
         "text": ("Frontend engineer joining Growth. React + TS + Tailwind, with "
                  "real focus on a11y and motion. We don't want a pixel-pusher; we "
                  "want a partner who'll push back."),
         "status": "approved", "hire_outcome": "good_hire"},
        {"id": "jd_past_ds_01", "role_family": "data_science",
         "text": ("Senior data scientist for the Insights team. You'll drive our "
                  "experimentation platform and partner with PMs on causal-inference "
                  "questions that don't have clean A/B answers."),
         "status": "approved", "hire_outcome": "good_hire"},
        {"id": "jd_past_pm_01", "role_family": "product",
         "text": ("Product manager for Platform. You'll own the developer surface — "
                  "everything from CLI ergonomics to API contracts. Strong written "
                  "communication; comfort with technical depth."),
         "status": "approved", "hire_outcome": "good_hire"},
        {"id": "jd_past_dz_01", "role_family": "design",
         "text": ("Senior product designer for the systems team. You'll evolve our "
                  "component library and partner with engineering on tokens. Comfort "
                  "moving between high-fidelity prototypes and production code review."),
         "status": "approved", "hire_outcome": "good_hire"},
    ]
    rejected = [
        # tone — jargon-heavy AI-sounding voice
        {"id": "jd_past_rej_01", "role_family": "backend",
         "text": ("We are seeking a 10x rockstar backend ninja to disrupt our "
                  "synergize cross-functional verticals leveraging cutting-edge "
                  "best-in-class technology stacks..."),
         "status": "rejected", "hire_outcome": None,
         "rejection": {
             "id": "rej_old_01",
             "text": ("Too much jargon — phrases like 'synergize cross-functional "
                      "verticals' don't match our straightforward voice."),
             "categories": ["tone"],
             "days_ago": 45,
         }},
        # bias — gender/age-coded language
        {"id": "jd_past_rej_02", "role_family": "frontend",
         "text": ("Looking for a young, energetic frontend ninja who is aggressive "
                  "about shipping and dominates the JavaScript ecosystem..."),
         "status": "rejected", "hire_outcome": None,
         "rejection": {
             "id": "rej_old_02",
             "text": ("Age-coded ('young, energetic') and gender-coded ('aggressive', "
                      "'dominate', 'ninja') language. Fails our inclusive-language bar."),
             "categories": ["bias", "tone"],
             "days_ago": 30,
         }},
        # requirements — unrealistic skill bundle
        {"id": "jd_past_rej_03", "role_family": "data_science",
         "text": ("Data scientist must have 10+ years PyTorch (released 2016), "
                  "PhD in CS from a top-5 institution, published at NeurIPS, plus "
                  "deep experience with Spark, Snowflake, dbt, Airflow, Kafka, "
                  "Flink, Beam, and Kubernetes admin..."),
         "status": "rejected", "hire_outcome": None,
         "rejection": {
             "id": "rej_old_03",
             "text": ("Unrealistic requirements — 10+ years on PyTorch is impossible "
                      "(it didn't exist), and the laundry list of tools screens out "
                      "exactly the experienced ICs we want."),
             "categories": ["requirements"],
             "days_ago": 14,
         }},
    ]

    out = []
    for jd in good:
        out.append({**jd, "created_at":
                    (today - timedelta(days=random.Random(SEED + 7).randint(60, 360))).isoformat()})
    for jd in rejected:
        days = jd["rejection"]["days_ago"]
        out.append({**jd, "created_at": (today - timedelta(days=days)).isoformat()})
    return out


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    catalog = build_skill_catalog()
    roles = build_roles()
    teams = build_teams()
    employees = build_employees(catalog)
    past_jds = build_past_jds()

    write_json(OUT_DIR / "esco_skills_subset.json", catalog)
    write_json(OUT_DIR / "synthetic_employees.json", {
        "_meta": {"seed": SEED, "generated_at": "2026-05-13",
                  "employee_count": len(employees),
                  "skill_count": len(catalog),
                  "role_count": len(roles),
                  "team_count": len(teams),
                  "past_jd_count": len(past_jds)},
        "skills":    catalog,
        "roles":     roles,
        "teams":     teams,
        "employees": employees,
        "past_jds":  past_jds,
    })
    print(f"wrote {OUT_DIR / 'esco_skills_subset.json'}  ({len(catalog)} skills)")
    print(f"wrote {OUT_DIR / 'synthetic_employees.json'}  ({len(employees)} employees, "
          f"{len(roles)} roles, {len(teams)} teams, {len(past_jds)} past JDs)")


if __name__ == "__main__":
    main()
