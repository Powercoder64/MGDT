"""Extracts clip-level features for Ego4D-NLQ with a trained Kinetics MGDT video encoder.

Videos are resampled to ``--target-fps``; feature ``j`` encodes ``--num-frames`` frames sampled
every ``--frame-stride`` frames starting at frame ``j * hop`` (hop = ``--hop``), so the feature
rate is ``target_fps / hop`` (1.875 features/s for 30 fps and hop 16). Each clip is saved as
``<output-dir>/<clip_uid>.pt`` holding a float16 tensor (T, C).

python tools/extract_video_features.py --config configs/kinetics400.yaml \
    --checkpoint outputs/kinetics400/final.pth --video-dir data/ego4d/v2/clips \
    --annotations data/ego4d/annotations/nlq_train.json data/ego4d/annotations/nlq_val.json \
    --output-dir data/ego4d/mgdt_features
"""

import argparse
import json
import os

import _bootstrap  # noqa: F401
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from mgdt.data.imagenet import IMAGENET_MEAN, IMAGENET_STD
from mgdt.engine.checkpoint import load_checkpoint, model_weights, strip_prefix
from mgdt.layers import set_schedule_progress
from mgdt.models import build_model
from mgdt.utils import load_config


def resampled_frames(path, target_fps, size):
    """Yields uint8 frames (3, size, size) on a uniform ``target_fps`` time grid."""
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = float(stream.average_rate) if stream.average_rate else target_fps
        next_slot = 0
        for count, frame in enumerate(container.decode(stream)):
            ts = frame.time if frame.time is not None else count / rate
            slot = int(round(ts * target_fps))
            if slot < next_slot:
                continue
            img = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
            img = TF.resize(img, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
            img = TF.center_crop(img, [size, size])
            while next_slot <= slot:
                yield img
                next_slot += 1


def normalise(clip):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)
    return (clip.float() / 255.0 - mean) / std


@torch.no_grad()
def extract(model, path, args, device):
    span = (args.num_frames - 1) * args.frame_stride
    buffer, base, window, batch, feats = [], 0, 0, [], []

    def flush():
        if batch:
            x = torch.stack(batch).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda"
                                else torch.bfloat16, enabled=args.amp):
                feats.append(model.forward_features(x).float().cpu())
            batch.clear()

    def take(start, total):
        idx = [min(start + i * args.frame_stride, total - 1) - base for i in range(args.num_frames)]
        return normalise(torch.stack([buffer[i] for i in idx], dim=1))

    total = 0
    for frame in resampled_frames(path, args.target_fps, args.size):
        buffer.append(frame)
        total += 1
        while window * args.hop + span <= total - 1:
            batch.append(take(window * args.hop, total))
            window += 1
            if len(batch) == args.batch_size:
                flush()
            drop = window * args.hop - base
            if drop > 0:
                del buffer[:drop]
                base += drop
    while window * args.hop < total:
        batch.append(take(window * args.hop, total))
        window += 1
        if len(batch) == args.batch_size:
            flush()
    flush()
    if not feats:
        raise RuntimeError(f"No frames decoded from {path}")
    return torch.cat(feats).half()


def clip_uids(annotation_files):
    uids = []
    for path in annotation_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        uids.extend(c["clip_uid"] for v in data["videos"] for c in v["clips"])
    return sorted(set(uids))


def main():
    parser = argparse.ArgumentParser(description="Extract MGDT clip features for Ego4D-NLQ")
    parser.add_argument("--config", required=True, help="video model config (e.g. configs/kinetics400.yaml)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-dir", required=True, help="folder with <clip_uid>.mp4 files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annotations", nargs="*", default=[], help="NLQ json files; restricts to their clips")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--hop", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-ema", action="store_true", help="use raw instead of EMA weights")
    parser.add_argument("--shard", type=int, nargs=2, default=(0, 1), metavar=("INDEX", "COUNT"),
                        help="process every COUNT-th clip starting at INDEX (for parallel jobs)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model)
    model.load_state_dict(strip_prefix(model_weights(load_checkpoint(args.checkpoint), not args.no_ema)))
    model.to(device).eval()
    set_schedule_progress(model, 1.0)

    if args.annotations:
        uids = clip_uids(args.annotations)
    else:
        uids = sorted(os.path.splitext(n)[0] for n in os.listdir(args.video_dir) if n.endswith(".mp4"))
    uids = uids[args.shard[0]::args.shard[1]]
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_fps": args.target_fps / args.hop, "hop": args.hop, "num_frames": args.num_frames,
                   "frame_stride": args.frame_stride, "target_fps": args.target_fps, "size": args.size}, f, indent=2)
    for i, uid in enumerate(uids):
        out = os.path.join(args.output_dir, uid + ".pt")
        if os.path.isfile(out):
            continue
        path = os.path.join(args.video_dir, uid + ".mp4")
        if not os.path.isfile(path):
            print(f"[{i + 1}/{len(uids)}] missing video {path}")
            continue
        feats = extract(model, path, args, device)
        torch.save(feats, out)
        print(f"[{i + 1}/{len(uids)}] {uid}: {tuple(feats.shape)}")


if __name__ == "__main__":
    main()
