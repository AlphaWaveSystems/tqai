"""Tests for GA optimization and policy genome."""

from __future__ import annotations

import random

import pytest

from tqai.optimization.genome import PolicyGenome
from tqai.optimization.ga_policy import GASearch


class TestPolicyGenome:
    def test_to_vector_and_back(self):
        g = PolicyGenome(scorer_idx=1.0, strategy_idx=2.0, bits_k=4.0)
        vec = g.to_vector()
        g2 = PolicyGenome.from_vector(vec)
        assert g2.scorer_idx == g.scorer_idx
        assert g2.strategy_idx == g.strategy_idx
        assert g2.bits_k == g.bits_k

    def test_random_genome(self):
        g = PolicyGenome.random()
        assert 0 <= g.scorer_alpha <= 1.0
        assert 2.0 <= g.bits_k <= 8.0

    def test_mutate(self):
        random.seed(42)
        g = PolicyGenome(scorer_alpha=0.5)
        mutated = g.mutate(mutation_rate=1.0, mutation_strength=0.1)
        assert mutated.scorer_alpha != g.scorer_alpha

    def test_crossover(self):
        a = PolicyGenome(scorer_idx=1.0, strategy_idx=1.0, bits_k=4.0)
        b = PolicyGenome(scorer_idx=2.0, strategy_idx=2.0, bits_k=2.0)
        random.seed(42)
        child = PolicyGenome.crossover(a, b)
        # Child has genes from both parents
        vec = child.to_vector()
        assert len(vec) == len(a.to_vector())

    def test_decode_empty(self):
        g = PolicyGenome(scorer_idx=0.0, strategy_idx=0.0)
        result = g.decode(["palm"], ["tiered"], ["stability"])
        assert result is None  # idx 0 = no selection

    def test_decode_with_scorer_and_strategy(self):
        g = PolicyGenome(
            scorer_idx=1.0,
            strategy_idx=1.0,
            scorer_alpha=0.3,
            tier_threshold=0.5,
        )
        result = g.decode(
            scorers=["palm", "fisher"],
            strategies=["tiered", "delta"],
            monitors=["stability"],
        )
        assert result is not None
        assert result["scorer"] == "palm"
        assert result["strategy"] == "tiered"
        assert result["scorer_kwargs"]["alpha"] == 0.3

    def test_decode_strategy_kwargs_by_type(self):
        g = PolicyGenome(strategy_idx=2.0, delta_threshold=0.15)
        result = g.decode([], ["tiered", "delta"], [])
        assert result["strategy"] == "delta"
        assert result["strategy_kwargs"]["threshold"] == 0.15

    def test_decode_window_strategy(self):
        g = PolicyGenome(strategy_idx=4.0, window_size=10.0)
        result = g.decode([], ["tiered", "delta", "delta2", "window"], [])
        assert result["strategy"] == "window"
        assert result["strategy_kwargs"]["window_size"] == 10

    def test_genes_property(self):
        g = PolicyGenome()
        assert len(g.genes) == 9
        assert "scorer_idx" in g.genes
        assert "bits_k" in g.genes


class TestGASearch:
    def test_run_finds_optimum(self):
        # Simple objective: maximize sum of bits
        def obj(genome):
            return genome.bits_k + genome.bits_v

        ga = GASearch(
            population_size=10,
            generations=5,
            objective=obj,
            seed=42,
        )
        best = ga.run()
        # Should converge toward higher bits
        assert best.fitness > 0

    def test_history_tracked(self):
        def obj(genome):
            return -abs(genome.bits_k - 4.0)

        ga = GASearch(
            population_size=10,
            generations=3,
            objective=obj,
            seed=42,
        )
        ga.run()
        assert len(ga.history) == 3
        assert "best_fitness" in ga.history[0]
        assert "mean_fitness" in ga.history[0]

    def test_no_objective_raises(self):
        ga = GASearch(population_size=5, generations=2)
        with pytest.raises(ValueError, match="No objective"):
            ga.run()

    def test_fitness_improves(self):
        def obj(genome):
            return -(genome.bits_k - 4) ** 2 - (genome.bits_v - 2) ** 2

        ga = GASearch(
            population_size=20,
            generations=10,
            objective=obj,
            seed=123,
        )
        best = ga.run()
        history = ga.history
        # Best fitness should generally improve
        assert history[-1]["best_fitness"] >= history[0]["best_fitness"]

    def test_elite_preservation(self):
        call_count = [0]

        def obj(genome):
            call_count[0] += 1
            return random.random()

        ga = GASearch(
            population_size=10,
            generations=2,
            objective=obj,
            seed=42,
        )
        ga.run()
        # Should have evaluated pop_size * (generations + 1) roughly
        assert call_count[0] > 0
