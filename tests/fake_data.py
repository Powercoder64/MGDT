"""Tiny synthetic datasets in the on-disk layouts expected by the data loaders."""

import csv
import json
import os

import numpy as np
import torch
from PIL import Image

WORDS = "a man woman dog cat sitting on the table with red blue car street riding near grass".split()


def _image(rng, w=48, h=40):
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def make_imagenet(root, classes=3, train_per_class=4, val_per_class=2, seed=0):
    rng = np.random.default_rng(seed)
    for split, n in (("train", train_per_class), ("val", val_per_class)):
        for c in range(classes):
            folder = os.path.join(root, split, f"n{c:08d}")
            os.makedirs(folder, exist_ok=True)
            for i in range(n):
                _image(rng).save(os.path.join(folder, f"img_{i}.JPEG"))
    return root


def write_video(path, frames, fps=30):
    import av

    T, H, W, _ = frames.shape
    with av.open(path, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def make_kinetics(root, classes=3, train_per_class=2, val_per_class=1, frames=12, seed=0):
    rng = np.random.default_rng(seed)
    videos = os.path.join(root, "videos")
    os.makedirs(videos, exist_ok=True)
    names = [f"class {c}" for c in range(classes)]
    annotations = {}
    for split, n in (("train", train_per_class), ("val", val_per_class)):
        rows = []
        for c in range(classes):
            for i in range(n):
                yid = f"{split[:2]}{c}{i}abcdefg"[:11]
                stem = f"{yid}_{i:06d}_{i + 10:06d}"
                write_video(os.path.join(videos, stem + ".mp4"),
                            rng.integers(0, 255, (frames, 40, 48, 3), dtype=np.uint8))
                rows.append({"label": names[c], "youtube_id": yid, "time_start": i, "time_end": i + 10, "split": split})
        ann = os.path.join(root, f"annotations_{split}.csv")
        with open(ann, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        annotations[split] = ann
    return annotations


def make_ade20k(root, train=4, val=2, seed=0):
    rng = np.random.default_rng(seed)
    for split, n in (("training", train), ("validation", val)):
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "annotations", split), exist_ok=True)
        for i in range(n):
            name = f"ADE_{split}_{i:08d}"
            _image(rng, 64, 48).convert("RGB").save(os.path.join(root, "images", split, name + ".jpg"))
            label = np.zeros((48, 64), dtype=np.uint8)
            label[:, :20] = 3
            label[:, 20:45] = 12
            label[10:30, 45:] = 150
            Image.fromarray(label).save(os.path.join(root, "annotations", split, name + ".png"))
    return root


def make_cityscapes(root, train=4, val=2, seed=0):
    rng = np.random.default_rng(seed)
    for split, n in (("train", train), ("val", val)):
        for i in range(n):
            city = "aachen" if i % 2 == 0 else "bremen"
            img_dir = os.path.join(root, "leftImg8bit", split, city)
            gt_dir = os.path.join(root, "gtFine", split, city)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)
            stem = f"{city}_{i:06d}_000019"
            _image(rng, 64, 32).save(os.path.join(img_dir, stem + "_leftImg8bit.png"))
            label = np.full((32, 64), 7, dtype=np.uint8)
            label[:12] = 23
            label[12:20, 10:40] = 26
            label[20:, 50:] = 0
            Image.fromarray(label).save(os.path.join(gt_dir, stem + "_gtFine_labelIds.png"))
    return root


def make_coco(root, seed=0):
    rng = np.random.default_rng(seed)
    images = []
    splits = ["train"] * 4 + ["restval"] * 2 + ["val"] * 2 + ["test"] * 3
    for i, split in enumerate(splits):
        folder = "train2014" if split in ("train", "restval") else "val2014"
        os.makedirs(os.path.join(root, "images", folder), exist_ok=True)
        name = f"COCO_{folder}_{i:012d}.jpg"
        _image(rng).save(os.path.join(root, "images", folder, name))
        sentences = [{"tokens": list(rng.choice(WORDS, rng.integers(4, 9))), "raw": ""} for _ in range(5)]
        images.append({"filepath": folder, "filename": name, "split": split, "cocoid": i, "sentences": sentences})
    path = os.path.join(root, "dataset_coco.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"images": images}, f)
    return path


def make_ego4d(root, feature_dim=16, fps=1.875, seed=0):
    rng = np.random.default_rng(seed)
    feat_dir = os.path.join(root, "features")
    os.makedirs(feat_dir, exist_ok=True)
    files = {}
    for split, clips in (("train", ["clipA", "clipB", "clipC"]), ("val", ["clipD", "clipE"])):
        videos = []
        for c, uid in enumerate(clips):
            duration = 60.0 + 10 * c
            torch.save(torch.randn(int(duration * fps), feature_dim).half(), os.path.join(feat_dir, uid + ".pt"))
            queries = []
            for q in range(4):
                start = float(rng.uniform(0, duration - 10))
                queries.append({"clip_start_sec": start, "clip_end_sec": start + float(rng.uniform(1, 8)),
                                "query": " ".join(rng.choice(WORDS, 5)) + "?"})
            queries.append({"clip_start_sec": 1.0, "clip_end_sec": 2.0})
            videos.append({"video_uid": f"video_{uid}", "clips": [{
                "clip_uid": uid, "video_start_sec": 100.0, "video_end_sec": 100.0 + duration,
                "annotations": [{"annotation_uid": f"ann_{uid}", "language_queries": queries}]}]})
        path = os.path.join(root, f"nlq_{split}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"videos": videos}, f)
        files[split] = path
    return files, feat_dir


def make_imdb(root, per_class=4, test_per_class=2, seed=0):
    rng = np.random.default_rng(seed)
    for split, n in (("train", per_class), ("test", test_per_class)):
        for label in ("pos", "neg"):
            folder = os.path.join(root, split, label)
            os.makedirs(folder, exist_ok=True)
            for i in range(n):
                text = " ".join(rng.choice(WORDS, rng.integers(5, 40))) + (" great!" if label == "pos" else " bad…")
                with open(os.path.join(folder, f"{i}_7.txt"), "w", encoding="utf-8") as f:
                    f.write(text)
    return root
