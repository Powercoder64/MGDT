"""Parameter count and inference FLOPs (multiply-accumulates) of a config's model.

FLOPs cover the full forward pass under the evaluation protocol: projections, sparse graph
construction, conductances, diffusion updates, restriction/prolongation, coarse corrections,
MLPs and task heads. Video FLOPs are per video (all test views); captioning FLOPs are per
caption (beam search for the maximum caption length).

python tools/count_flops.py --config configs/imagenet1k.yaml
"""

import argparse
import json

import _bootstrap  # noqa: F401
import torch

from mgdt.layers import set_schedule_progress
from mgdt.models import build_model
from mgdt.utils import count_parameters, load_config, seed_everything
from mgdt.utils.flops import count_macs


def _measure(fn):
    with torch.no_grad(), count_macs() as result:
        fn()
    return result["total"]


def _caption_macs(model, cfg):
    test = cfg.data.get("test", {})
    beam, words = int(test.get("beam_size", 3)), int(test.get("max_words", 20))
    img = torch.randn(1, 3, int(cfg.data.img_size), int(cfg.data.img_size))
    state = {}
    total = _measure(lambda: state.update(start=model.start_decoding(img, beam)))
    memory_kv, caches = state["start"]
    tokens = torch.full((beam,), model.bos_id, dtype=torch.long)
    for t in range(min(words + 1, model.max_len)):
        total += _measure(lambda: model.decode_step(tokens, t, memory_kv, caches))
    return total


def main():
    parser = argparse.ArgumentParser(description="Count MGDT parameters and FLOPs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--vocab-size", type=int, default=9487, help="caption/query vocabulary size")
    parser.add_argument("--video-seconds", type=float, default=498.0,
                        help="average Ego4D-NLQ clip duration used for the grounding head")
    parser.add_argument("--query-words", type=int, default=12, help="average NLQ query length")
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.opts)
    seed_everything(0)
    task = cfg.task
    extra = {}
    if task == "captioning":
        extra = {"vocab_size": args.vocab_size}
    elif task == "grounding":
        extra = {"vocab_size": args.vocab_size, "video_dim": int(cfg.data.feature_dim)}
    elif task == "segmentation":
        extra = {"num_classes": 150 if cfg.data.dataset == "ade20k" else 19}
    model = build_model(cfg.model, **extra).eval()
    set_schedule_progress(model, 1.0)
    report = {"config": args.config, "params_M": count_parameters(model) / 1e6}

    if task == "imagenet":
        s = int(cfg.data.img_size)
        x = torch.randn(1, 3, s, s)
        report["GFLOPs_per_image"] = _measure(lambda: model(x)) / 1e9
    elif task == "kinetics":
        d = cfg.data
        x = torch.randn(1, 3, int(d.num_frames), int(d.img_size), int(d.img_size))
        per_view = _measure(lambda: model(x))
        views = int(d.get("test_clips", 4)) * int(d.get("test_crops", 3))
        report["GFLOPs_per_view"] = per_view / 1e9
        report["TFLOPs_per_video"] = per_view * views / 1e12
        report["views"] = views
    elif task == "segmentation":
        test = cfg.data.get("test", {})
        if test.get("mode", "whole") == "slide":
            h, w = test.crop_size
            report["input"] = f"{h}x{w} crop"
        else:
            h, w = (1024, 2048)
            report["input"] = f"{h}x{w} full resolution"
        x = torch.randn(1, 3, int(h), int(w))
        report["GFLOPs"] = _measure(lambda: model(x)) / 1e9
    elif task == "captioning":
        report["GFLOPs_per_caption"] = _caption_macs(model, cfg) / 1e9
    elif task == "grounding":
        T = int(round(args.video_seconds * float(cfg.data.feature_fps)))
        video = torch.randn(1, T, int(cfg.data.feature_dim))
        vmask = torch.ones(1, T, dtype=torch.bool)
        words = torch.randint(4, args.vocab_size, (1, args.query_words))
        wmask = torch.ones_like(words, dtype=torch.bool)
        report["feature_steps"] = T
        report["GFLOPs_grounding_head"] = _measure(lambda: model(video, vmask, words, wmask)) / 1e9
    elif task == "text":
        L = int(cfg.data.max_len)
        tokens = torch.randint(1, 257, (1, L))
        report["GFLOPs_per_sequence"] = _measure(lambda: model(tokens)) / 1e9
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
