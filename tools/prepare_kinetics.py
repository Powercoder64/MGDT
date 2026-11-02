"""Builds ``path,label`` lists for Kinetics-400/700 from the official annotation CSVs.

Videos are matched by file stem ``<youtube_id>_<start:06d>_<end:06d>`` (the naming used by the
public Kinetics downloads) or, failing that, by a stem starting with the YouTube id.

python tools/prepare_kinetics.py --annotations data/kinetics400/annotations/train.csv \
    --video-root data/kinetics400/videos --output data/kinetics400/train.csv \
    --classes data/kinetics400/classes.txt
"""

import argparse
import csv
import os

import _bootstrap  # noqa: F401

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def index_videos(root):
    """Maps file stems to paths relative to ``root``."""
    index = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(VIDEO_EXTS):
                stem = os.path.splitext(name)[0]
                index[stem] = os.path.relpath(os.path.join(dirpath, name), root)
    return index


def read_annotations(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Create Kinetics video lists")
    parser.add_argument("--annotations", required=True, help="official CSV (label,youtube_id,time_start,time_end,...)")
    parser.add_argument("--video-root", required=True, help="folder containing the downloaded clips")
    parser.add_argument("--output", required=True, help="output list (path,label)")
    parser.add_argument("--classes", required=True,
                        help="class-name file; created from these annotations if it does not exist")
    parser.add_argument("--check", action="store_true", help="drop videos that cannot be decoded (slow)")
    args = parser.parse_args()

    rows = read_annotations(args.annotations)
    if os.path.isfile(args.classes):
        with open(args.classes, "r", encoding="utf-8") as f:
            classes = [line.rstrip("\n") for line in f if line.strip()]
    else:
        classes = sorted({r["label"] for r in rows})
        os.makedirs(os.path.dirname(os.path.abspath(args.classes)), exist_ok=True)
        with open(args.classes, "w", encoding="utf-8") as f:
            f.write("\n".join(classes) + "\n")
    class_index = {c: i for i, c in enumerate(classes)}
    videos = index_videos(args.video_root)
    by_id = {}
    for stem, rel in videos.items():
        by_id.setdefault(stem[:11], rel)

    entries, missing, unknown, broken = [], 0, 0, 0
    for r in rows:
        label = r.get("label")
        if label not in class_index:
            unknown += 1
            continue
        yid = r["youtube_id"]
        stem = f"{yid}_{int(float(r['time_start'])):06d}_{int(float(r['time_end'])):06d}"
        rel = videos.get(stem) or by_id.get(yid)
        if rel is None:
            missing += 1
            continue
        if args.check:
            from mgdt.data.video import decode_video

            try:
                decode_video(os.path.join(args.video_root, rel))
            except Exception:
                broken += 1
                continue
        entries.append((rel, class_index[label]))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(entries)
    print(f"Wrote {len(entries)} videos to {args.output} ({len(classes)} classes); "
          f"missing {missing}, unknown labels {unknown}, undecodable {broken}")


if __name__ == "__main__":
    main()
