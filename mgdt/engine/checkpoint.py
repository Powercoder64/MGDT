"""Checkpoint saving and loading."""

import os

import torch


def unwrap(model):
    """Returns the module inside DistributedDataParallel, if any."""
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path, model, optimizer=None, scaler=None, ema=None, step=0, epoch=0, cfg=None, metrics=None):
    """Writes model/optimiser/scaler/EMA state with progress counters (plain types only)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state = {
        "model": unwrap(model).state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "cfg": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg,
        "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in (metrics or {}).items()},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    """Loads a checkpoint dictionary onto the CPU."""
    return torch.load(path, map_location="cpu", weights_only=True)


def model_weights(checkpoint, prefer_ema=True):
    """Model state dict from a checkpoint, using EMA weights when available and requested."""
    if prefer_ema and checkpoint.get("ema"):
        return checkpoint["ema"]
    return checkpoint["model"] if "model" in checkpoint else checkpoint


def strip_prefix(state_dict, prefixes=("module.",)):
    """Removes wrapper prefixes from state-dict keys."""
    out = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
        out[key] = value
    return out
