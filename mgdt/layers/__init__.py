import torch.nn as nn

from ..ops.conductance import Conductance
from .attention import CrossAttention, SoftmaxAttention, scaled_dot_product
from .block import Block, build_mixer
from .common import DropPath, Mlp, init_weights, sinusoidal_positions
from .mgdt_mixer import MGDTMixer


def set_schedule_progress(model: nn.Module, progress: float):
    """Sets the conductance warm-up progress of every MGDT mixer in ``model``."""
    for module in model.modules():
        if isinstance(module, Conductance):
            module.set_progress(progress)


def set_energy_tracking(model: nn.Module, enabled: bool):
    """Turns diffusion-energy tracking on or off for every MGDT mixer in ``model``."""
    for module in model.modules():
        if isinstance(module, MGDTMixer):
            module.track_energy = bool(enabled)
            module.last_energy_reduction = None


def mean_energy_reduction(model: nn.Module):
    """Average per-cycle diffusion-energy reduction over MGDT mixers from the last forward pass."""
    values = [m.last_energy_reduction for m in model.modules()
              if isinstance(m, MGDTMixer) and m.last_energy_reduction is not None]
    return sum(values) / len(values) if values else None


__all__ = [
    "Block",
    "CrossAttention",
    "DropPath",
    "MGDTMixer",
    "Mlp",
    "SoftmaxAttention",
    "build_mixer",
    "init_weights",
    "mean_energy_reduction",
    "scaled_dot_product",
    "set_energy_tracking",
    "set_schedule_progress",
    "sinusoidal_positions",
]
