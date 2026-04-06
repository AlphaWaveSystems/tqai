"""Policy genome definition for GA-based pipeline optimization.

A genome encodes a complete pipeline configuration as a flat vector
of floats that can be mutated and crossed over by the GA.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolicyGenome:
    """Encodes a pipeline configuration as an evolvable genome.

    Each gene maps to a pipeline parameter.  The genome can be decoded
    into a ``pipeline`` config dict for ``TurboQuantConfig``.

    Genes:
        - scorer_idx: Index into available scorers (0 = none)
        - strategy_idx: Index into available strategies (0 = none)
        - monitor_idx: Index into available monitors (0 = none)
        - scorer_alpha: EMA decay for scorer (0.01 - 1.0)
        - tier_threshold: High-tier threshold (0.1 - 0.9)
        - delta_threshold: Delta strategy threshold (0.01 - 0.5)
        - window_size: Window strategy size (1 - 20)
        - bits_k: Key bits (2 - 8)
        - bits_v: Value bits (2 - 8)
    """

    scorer_idx: float = 0.0
    strategy_idx: float = 0.0
    monitor_idx: float = 0.0
    scorer_alpha: float = 0.5
    tier_threshold: float = 0.4
    delta_threshold: float = 0.1
    window_size: float = 5.0
    bits_k: float = 4.0
    bits_v: float = 2.0
    fitness: float = 0.0

    _GENE_NAMES: list[str] = field(default=None, repr=False, init=False)

    def __post_init__(self):
        self._GENE_NAMES = [
            "scorer_idx", "strategy_idx", "monitor_idx",
            "scorer_alpha", "tier_threshold", "delta_threshold",
            "window_size", "bits_k", "bits_v",
        ]

    @property
    def genes(self) -> list[str]:
        return list(self._GENE_NAMES)

    def to_vector(self) -> list[float]:
        return [getattr(self, g) for g in self._GENE_NAMES]

    @classmethod
    def from_vector(cls, vec: list[float]) -> PolicyGenome:
        names = [
            "scorer_idx", "strategy_idx", "monitor_idx",
            "scorer_alpha", "tier_threshold", "delta_threshold",
            "window_size", "bits_k", "bits_v",
        ]
        kwargs = {n: v for n, v in zip(names, vec)}
        return cls(**kwargs)

    def decode(
        self,
        scorers: list[str],
        strategies: list[str],
        monitors: list[str],
    ) -> dict[str, Any]:
        """Decode genome into a pipeline config dict.

        Args:
            scorers: Available scorer names (from registry).
            strategies: Available strategy names.
            monitors: Available monitor names.

        Returns:
            Dict suitable for ``TurboQuantConfig.pipeline``.
        """
        config: dict[str, Any] = {}

        s_idx = int(self.scorer_idx) % (len(scorers) + 1)
        if s_idx > 0 and scorers:
            config["scorer"] = scorers[s_idx - 1]
            config["scorer_kwargs"] = {"alpha": max(0.01, min(1.0, self.scorer_alpha))}

        st_idx = int(self.strategy_idx) % (len(strategies) + 1)
        if st_idx > 0 and strategies:
            name = strategies[st_idx - 1]
            config["strategy"] = name
            if name == "tiered":
                config["strategy_kwargs"] = {
                    "high_tier_threshold": max(0.1, min(0.9, self.tier_threshold))
                }
            elif name in ("delta", "delta2"):
                config["strategy_kwargs"] = {
                    "threshold": max(0.01, min(0.5, self.delta_threshold))
                }
            elif name == "window":
                config["strategy_kwargs"] = {
                    "window_size": max(1, int(self.window_size))
                }

        m_idx = int(self.monitor_idx) % (len(monitors) + 1)
        if m_idx > 0 and monitors:
            config["monitor"] = monitors[m_idx - 1]

        return config if config else None

    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.3) -> PolicyGenome:
        """Return a mutated copy."""
        vec = self.to_vector()
        new_vec = []
        for v in vec:
            if random.random() < mutation_rate:
                v += random.gauss(0, mutation_strength)
            new_vec.append(v)
        return PolicyGenome.from_vector(new_vec)

    @staticmethod
    def crossover(a: PolicyGenome, b: PolicyGenome) -> PolicyGenome:
        """Single-point crossover."""
        va = a.to_vector()
        vb = b.to_vector()
        point = random.randint(1, len(va) - 1)
        child_vec = va[:point] + vb[point:]
        return PolicyGenome.from_vector(child_vec)

    @staticmethod
    def random() -> PolicyGenome:
        """Generate a random genome."""
        return PolicyGenome(
            scorer_idx=random.uniform(0, 5),
            strategy_idx=random.uniform(0, 5),
            monitor_idx=random.uniform(0, 3),
            scorer_alpha=random.uniform(0.01, 1.0),
            tier_threshold=random.uniform(0.1, 0.9),
            delta_threshold=random.uniform(0.01, 0.5),
            window_size=random.uniform(1, 20),
            bits_k=random.uniform(2, 8),
            bits_v=random.uniform(2, 8),
        )
