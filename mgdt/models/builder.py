"""Model construction from the ``model`` section of a config."""

import copy

from .captioning import MGDTCaptioner
from .grounding import MGDTGroundingHead
from .segmentation import MGDTSegmenter
from .text import MGDTTextClassifier
from .video import MGDTVideoTransformer
from .vit import MGDTVisionTransformer

MODELS = {
    "mgdt_vit": MGDTVisionTransformer,
    "mgdt_video": MGDTVideoTransformer,
    "mgdt_segmenter": MGDTSegmenter,
    "mgdt_captioner": MGDTCaptioner,
    "mgdt_grounding": MGDTGroundingHead,
    "mgdt_text": MGDTTextClassifier,
}


def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def build_model(model_cfg, **extra):
    """Instantiates ``MODELS[model_cfg['name']]`` with the remaining keys as arguments."""
    cfg = copy.deepcopy(_plain(dict(model_cfg)))
    name = cfg.pop("name")
    cfg.pop("pretrained", None)
    cfg.pop("pretrained_ema", None)
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    cfg.update(extra)
    return MODELS[name](**cfg)
