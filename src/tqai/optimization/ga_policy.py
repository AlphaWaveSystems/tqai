"""Genetic algorithm policy search over pipeline configurations.

Evolves a population of ``PolicyGenome`` instances to find the Pareto-
optimal pipeline configuration for a given model and objective (e.g.,
minimize perplexity degradation while maximizing compression).

Usage::

    from tqai.optimization import GASearch

    search = GASearch(
        population_size=20,
        generations=10,
        objective=lambda genome: evaluate(genome),
    )
    best = search.run()
    print(best.decode(scorers, strategies, monitors))
"""

from __future__ import annotations

import random
from typing import Any, Callable

from tqai.optimization.genome import PolicyGenome


class GASearch:
    """Genetic algorithm search over pipeline configurations.

    Args:
        population_size: Number of genomes per generation.
        generations: Number of evolution generations.
        objective: Callable that takes a ``PolicyGenome`` and returns
            a fitness score (higher is better).
        mutation_rate: Probability of mutating each gene (default 0.15).
        mutation_strength: Standard deviation of Gaussian mutation.
        elite_fraction: Top fraction to carry forward unchanged.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        objective: Callable[[PolicyGenome], float] | None = None,
        mutation_rate: float = 0.15,
        mutation_strength: float = 0.3,
        elite_fraction: float = 0.2,
        seed: int | None = None,
    ):
        self._pop_size = population_size
        self._generations = generations
        self._objective = objective
        self._mutation_rate = mutation_rate
        self._mutation_strength = mutation_strength
        self._elite_count = max(1, int(population_size * elite_fraction))
        self._history: list[dict[str, Any]] = []

        if seed is not None:
            random.seed(seed)

    def run(self) -> PolicyGenome:
        """Run the GA and return the best genome."""
        if self._objective is None:
            raise ValueError("No objective function provided")

        population = [PolicyGenome.random() for _ in range(self._pop_size)]

        for gen in range(self._generations):
            # Evaluate
            for genome in population:
                genome.fitness = self._objective(genome)

            # Sort by fitness (descending)
            population.sort(key=lambda g: g.fitness, reverse=True)

            self._history.append({
                "generation": gen,
                "best_fitness": population[0].fitness,
                "mean_fitness": sum(g.fitness for g in population) / len(population),
                "best_genome": population[0].to_vector(),
            })

            # Select parents (elitism + tournament)
            elites = population[: self._elite_count]
            new_pop = list(elites)

            while len(new_pop) < self._pop_size:
                a = self._tournament_select(population)
                b = self._tournament_select(population)
                child = PolicyGenome.crossover(a, b)
                child = child.mutate(self._mutation_rate, self._mutation_strength)
                new_pop.append(child)

            population = new_pop

        # Final eval
        for genome in population:
            genome.fitness = self._objective(genome)

        population.sort(key=lambda g: g.fitness, reverse=True)
        return population[0]

    def _tournament_select(self, population: list[PolicyGenome], k: int = 3) -> PolicyGenome:
        contenders = random.sample(population, min(k, len(population)))
        return max(contenders, key=lambda g: g.fitness)

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)
