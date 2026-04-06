"""Name-based registry for pipeline middleware components."""

from __future__ import annotations

from typing import Any

_SCORERS: dict[str, type] = {}
_STRATEGIES: dict[str, type] = {}
_MONITORS: dict[str, type] = {}
_ADAPTERS: dict[str, type] = {}


# -- Registration ----------------------------------------------------------


def register_scorer(name: str, cls: type) -> None:
    _SCORERS[name] = cls


def register_strategy(name: str, cls: type) -> None:
    _STRATEGIES[name] = cls


def register_monitor(name: str, cls: type) -> None:
    _MONITORS[name] = cls


def register_adapter(name: str, cls: type) -> None:
    _ADAPTERS[name] = cls


# -- Lookup ----------------------------------------------------------------


def get_scorer(name: str, **kwargs: Any) -> Any:
    if name not in _SCORERS:
        raise ValueError(f"Unknown scorer: {name!r}. Available: {list(_SCORERS)}")
    return _SCORERS[name](**kwargs)


def get_strategy(name: str, **kwargs: Any) -> Any:
    if name not in _STRATEGIES:
        raise ValueError(f"Unknown strategy: {name!r}. Available: {list(_STRATEGIES)}")
    return _STRATEGIES[name](**kwargs)


def get_monitor(name: str, **kwargs: Any) -> Any:
    if name not in _MONITORS:
        raise ValueError(f"Unknown monitor: {name!r}. Available: {list(_MONITORS)}")
    return _MONITORS[name](**kwargs)


def get_adapter(name: str | None, model: Any) -> Any:
    if name is not None:
        if name not in _ADAPTERS:
            raise ValueError(
                f"Unknown adapter: {name!r}. Available: {list(_ADAPTERS)}"
            )
        return _ADAPTERS[name]()
    # Auto-detect: try each adapter
    for adapter_cls in _ADAPTERS.values():
        adapter = adapter_cls()
        if adapter.detect(model):
            return adapter
    raise ValueError(f"No adapter found for model type: {type(model)}")


# -- Introspection ---------------------------------------------------------


def list_available() -> dict[str, list[str]]:
    return {
        "scorers": list(_SCORERS),
        "strategies": list(_STRATEGIES),
        "monitors": list(_MONITORS),
        "adapters": list(_ADAPTERS),
    }
