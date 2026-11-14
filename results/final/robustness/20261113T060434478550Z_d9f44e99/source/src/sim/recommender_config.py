"""Resolve shared and per-algorithm configuration for registered recommenders."""
from inspect import signature

from .recommenders import REGISTRY


def recommender_options(name, settings, **overrides):
    allowed = set(signature(REGISTRY[name].__init__).parameters) - {"self"}
    options = {key: value for key, value in settings.items() if key in allowed}
    parent = {"fatigue_mf": "mf", "fatigue_content": "content",
              "fatigue_popularity": "popularity"}.get(name)
    inherited = settings.get(parent, {}) if parent else {}
    specific = settings.get(name, {})
    if not isinstance(inherited, dict):
        raise ValueError(f"recommenders.{parent} must be a mapping")
    if not isinstance(specific, dict):
        raise ValueError(f"recommenders.{name} must be a mapping")
    specific = {**inherited, **specific}
    unknown = (set(specific) | set(overrides)) - allowed
    if unknown:
        raise ValueError(f"Unknown settings for {name}: {sorted(unknown)}")
    return {**options, **specific, **overrides}
