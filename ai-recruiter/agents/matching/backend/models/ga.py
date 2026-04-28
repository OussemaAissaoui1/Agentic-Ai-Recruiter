"""Constraint-Aware Genetic Algorithm (CAGA).

Implementation of Malini et al. 2026 "Optimizing Candidate-Job Fit Using a
Constraint-Aware Genetic Algorithm Framework" (ETASR 16(1) 13543).

Faithful to the paper's equations:
    Eq. 1-2  per-candidate fit score = weighted sum of 5 attributes
    Eq. 3    chromosome fitness = sum_i z_i * FitScore_i  (maximise)
    Eq. 4    budget      sum_i z_i * cost_i   <= B
    Eq. 5    diversity   sum_i z_i * g_i / sum_i z_i  >= delta
    Eq. 6    role        sum_{i:r_i=k} z_i  >= R_k  for each role k
    Eq. 7-8  min fit     z_i = 1 implies FitScore_i >= tau
    Eq. 9    weights normalise to 1

Enforcement is death-penalty: if any hard constraint is violated the
chromosome's fitness is zeroed (pseudocode Step 5-7).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import GA_CFG


@dataclass
class GACandidate:
    """One candidate's attributes as they enter the GA.

    The 5 attribute scores must be in [0, 1] (the paper normalises them).
    cost, gender and role come from the recruiter form.
    """

    id: str
    skills_match: float        # s_i
    interview_score: float     # I_i
    cultural_fit: float        # c_i
    experience_score: float    # e_i
    salary_alignment: float    # p_i
    cost: float                # c_i in Eq. 4 (currency)
    gender_female: int         # g_i in Eq. 5: 1 if female, else 0
    role: str                  # r_i in Eq. 6


@dataclass
class GAConstraints:
    budget: float                                  # B in Eq. 4
    min_female_ratio: float                        # delta in Eq. 5
    role_requirements: Dict[str, int] = field(default_factory=dict)  # R_k
    min_fit_threshold: float = 0.5                 # tau in Eq. 7


class ConstraintAwareGA:
    def __init__(
        self,
        candidates: List[GACandidate],
        constraints: GAConstraints,
        weights: Optional[Tuple[float, float, float, float, float]] = None,
        population_size: int = GA_CFG.population_size,
        num_generations: int = GA_CFG.num_generations,
        crossover_rate: float = GA_CFG.crossover_rate,
        mutation_rate: Optional[float] = None,
        tournament_k: int = GA_CFG.tournament_k,
        elitism: int = GA_CFG.elitism,
        seed: int = 42,
    ) -> None:
        self.candidates = candidates
        self.m = len(candidates)
        self.constraints = constraints
        if weights is None:
            weights = (
                GA_CFG.w_skills,
                GA_CFG.w_interview,
                GA_CFG.w_cultural,
                GA_CFG.w_experience,
                GA_CFG.w_salary,
            )
        total = sum(weights)
        if total <= 0:
            raise ValueError("Weight vector must have positive sum (Eq. 9).")
        self.weights = tuple(w / total for w in weights)

        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        # Paper does not fix mutation rate; canonical default is 1/m.
        self.mutation_rate = mutation_rate if mutation_rate is not None else max(1.0 / max(1, self.m), 1e-3)
        self.tournament_k = tournament_k
        self.elitism = elitism

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self._fit_scores = np.array([self._compute_fit_score(c) for c in candidates], dtype=np.float64)
        self._costs = np.array([c.cost for c in candidates], dtype=np.float64)
        self._gender = np.array([c.gender_female for c in candidates], dtype=np.float64)
        self._roles = [c.role for c in candidates]

        self.history: List[float] = []
        self.best_z: Optional[np.ndarray] = None
        self.best_fitness: float = 0.0

    # ------------------------------------------------------------- fitness
    def _compute_fit_score(self, c: GACandidate) -> float:
        w_s, w_i, w_c, w_e, w_p = self.weights
        return (
            w_s * c.skills_match
            + w_i * c.interview_score
            + w_c * c.cultural_fit
            + w_e * c.experience_score
            + w_p * c.salary_alignment
        )

    def _feasible(self, z: np.ndarray) -> bool:
        sel = z.astype(bool)
        total_cost = float(self._costs[sel].sum())
        if total_cost > self.constraints.budget:
            return False
        if sel.sum() > 0:
            female_ratio = float(self._gender[sel].sum() / sel.sum())
            if female_ratio < self.constraints.min_female_ratio:
                return False
        for role, required in self.constraints.role_requirements.items():
            cnt = sum(1 for i in np.where(sel)[0] if self._roles[i] == role)
            if cnt < required:
                return False
        # Eq. 7 / 8: every selected candidate must clear tau
        for i in np.where(sel)[0]:
            if self._fit_scores[i] < self.constraints.min_fit_threshold:
                return False
        return True

    def _fitness(self, z: np.ndarray) -> float:
        if z.sum() == 0:
            return 0.0
        if not self._feasible(z):
            return 0.0
        return float((self._fit_scores * z).sum())

    # -------------------------------------------------------------- genetics
    def _tournament(self, pop: List[np.ndarray], fits: List[float]) -> np.ndarray:
        idx = self.rng.sample(range(len(pop)), self.tournament_k)
        best = max(idx, key=lambda i: fits[i])
        return pop[best].copy()

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.crossover_rate or self.m < 2:
            return a.copy(), b.copy()
        # Two-point crossover
        p1, p2 = sorted(self.rng.sample(range(self.m), 2))
        c1 = a.copy(); c2 = b.copy()
        c1[p1:p2] = b[p1:p2]
        c2[p1:p2] = a[p1:p2]
        return c1, c2

    def _mutate(self, z: np.ndarray) -> np.ndarray:
        flip = self.np_rng.random(self.m) < self.mutation_rate
        z[flip] = 1 - z[flip]
        return z

    # --------------------------------------------------------------- driver
    def run(self) -> Tuple[np.ndarray, float, List[float]]:
        # Initial random population
        pop = [self.np_rng.integers(0, 2, size=self.m) for _ in range(self.population_size)]
        fits = [self._fitness(z) for z in pop]

        for _ in range(self.num_generations):
            # Rank and record best-so-far
            order = np.argsort(fits)[::-1]
            if fits[order[0]] > self.best_fitness:
                self.best_fitness = float(fits[order[0]])
                self.best_z = pop[order[0]].copy()
            self.history.append(self.best_fitness)

            # Elitism
            new_pop: List[np.ndarray] = [pop[order[i]].copy() for i in range(min(self.elitism, len(pop)))]
            while len(new_pop) < self.population_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1); c2 = self._mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < self.population_size:
                    new_pop.append(c2)
            pop = new_pop
            fits = [self._fitness(z) for z in pop]

        # Last generation check
        order = np.argsort(fits)[::-1]
        if fits[order[0]] > self.best_fitness:
            self.best_fitness = float(fits[order[0]])
            self.best_z = pop[order[0]].copy()
        self.history.append(self.best_fitness)

        if self.best_z is None:
            self.best_z = np.zeros(self.m, dtype=np.int64)
        return self.best_z, self.best_fitness, self.history

    def selected_candidates(self) -> List[GACandidate]:
        if self.best_z is None:
            return []
        return [self.candidates[i] for i in np.where(self.best_z == 1)[0]]
