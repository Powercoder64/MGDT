"""YAML configuration loading with base-file inheritance and command-line overrides."""

import copy
import os

import yaml


class Config(dict):
    """Dictionary with attribute access to keys."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def to_dict(self):
        """Returns a plain nested dict copy."""
        return _to_plain(self)


def _to_config(obj):
    if isinstance(obj, dict):
        return Config({k: _to_config(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config(v) for v in obj]
    return obj


def _to_plain(obj):
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def merge_dicts(base, override):
    """Recursively merges ``override`` into a copy of ``base``; ``_replace_: true`` replaces a sub-dict."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and value.get("_replace_", False):
            out[key] = {k: v for k, v in value.items() if k != "_replace_"}
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dicts(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    bases = data.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged = {}
    for base in bases:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        merged = merge_dicts(merged, _load_yaml(base_path))
    return merge_dicts(merged, data)


def set_by_path(cfg, dotted_key, value):
    """Sets ``cfg[a][b][c] = value`` for ``dotted_key = 'a.b.c'``, creating dicts as needed."""
    keys = dotted_key.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def apply_overrides(cfg, overrides):
    """Applies ``key=value`` strings (values parsed as YAML) to a nested dict in place."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must have the form key=value")
        key, raw = item.split("=", 1)
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def load_config(path, overrides=None):
    """Loads a YAML config with ``_base_`` inheritance and applies ``key=value`` overrides."""
    cfg = _load_yaml(path)
    apply_overrides(cfg, overrides)
    return _to_config(cfg)
