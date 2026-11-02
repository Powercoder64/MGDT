from .base import Task, move_to
from .captioning import CaptioningTask
from .classification import ImageNetTask
from .grounding import GroundingTask, decode_spans
from .segmentation import SegmentationTask, slide_inference
from .text import TextClassificationTask
from .video import KineticsTask

TASKS = {
    "imagenet": ImageNetTask,
    "kinetics": KineticsTask,
    "segmentation": SegmentationTask,
    "captioning": CaptioningTask,
    "grounding": GroundingTask,
    "text": TextClassificationTask,
}


def build_task(cfg, device, dist_info, logger):
    """Instantiates the task named by ``cfg.task``."""
    if cfg.task not in TASKS:
        raise ValueError(f"Unknown task '{cfg.task}'. Available: {sorted(TASKS)}")
    return TASKS[cfg.task](cfg, device, dist_info, logger)


__all__ = ["TASKS", "Task", "build_task", "decode_spans", "move_to", "slide_inference"]
