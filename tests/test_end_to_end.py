"""Trains and evaluates every benchmark pipeline for a few steps on synthetic data (CPU)."""

import glob
import json
import os
import subprocess
import sys

import pytest
import torch

from fake_data import (make_ade20k, make_cityscapes, make_coco, make_ego4d, make_imagenet, make_imdb,
                       make_kinetics)
from mgdt.engine import evaluate, train
from mgdt.utils import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY_BACKBONE = ["img_size=32", "patch_size=8", "embed_dim=32", "depth=2", "num_heads=2", "mixer.k=4"]
COMMON = ["train.amp=none", "data.num_workers=0", "train.log_interval=1"]


def cfg_for(name, out, opts, backbone_prefix="model."):
    overrides = COMMON + [backbone_prefix + o for o in TINY_BACKBONE] + opts + [f"output_dir={out}"]
    return load_config(os.path.join(ROOT, "configs", f"{name}.yaml"), overrides)


def run_and_check(cfg, metric):
    results = train(cfg)
    assert not results["diverged"]
    assert metric in results
    assert os.path.isfile(os.path.join(cfg.output_dir, "final.pth"))
    with open(os.path.join(cfg.output_dir, "results.json"), encoding="utf-8") as f:
        assert metric in json.load(f)
    metrics = evaluate(cfg, os.path.join(cfg.output_dir, "final.pth"))
    assert metric in metrics
    return results


@pytest.fixture(scope="session")
def imagenet_ckpt(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("imagenet")
    root = make_imagenet(str(tmp / "data"))
    cfg = cfg_for("imagenet1k", str(tmp / "out"), [
        f"data.root={root}", "data.img_size=32", "model.num_classes=3", "schedule.epochs=2",
        "schedule.warmup_epochs=1", "schedule.kappa_warmup_epochs=1", "train.batch_size=4",
        "train.eval_batch_size=4", "train.ema.decay=0.9"])
    results = run_and_check(cfg, "top1")
    assert "ema_top1" in results
    return os.path.join(cfg.output_dir, "final.pth")


def test_imagenet(imagenet_ckpt):
    state = torch.load(imagenet_ckpt, map_location="cpu", weights_only=True)
    assert "ema" in state and "model" in state


def test_imagenet_resume(imagenet_ckpt, tmp_path):
    out = os.path.dirname(imagenet_ckpt)
    cfg = load_config(os.path.join(out, "config.yaml"), ["schedule.epochs=3"])
    results = train(cfg)
    assert results["epoch"] == 3 and not results["diverged"]


def test_kinetics_and_feature_extraction(imagenet_ckpt, tmp_path):
    ann = make_kinetics(str(tmp_path / "k400"))
    lists = {}
    for split in ("train", "val"):
        lists[split] = str(tmp_path / "k400" / f"{split}.csv")
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "prepare_kinetics.py"), "--annotations", ann[split],
                        "--video-root", str(tmp_path / "k400" / "videos"), "--output", lists[split],
                        "--classes", str(tmp_path / "k400" / "classes.txt"), "--check"], check=True)
    cfg = cfg_for("kinetics400", str(tmp_path / "out"), [
        f"data.root={tmp_path / 'k400' / 'videos'}", f"data.train_list={lists['train']}",
        f"data.val_list={lists['val']}", "data.num_frames=4", "data.img_size=32", "data.test_clips=2",
        "model.num_frames=4", "model.temporal_depth=1", "model.num_classes=3", f"model.pretrained={imagenet_ckpt}",
        "schedule.epochs=2", "schedule.warmup_epochs=1", "schedule.kappa_warmup_epochs=1", "train.batch_size=4",
        "train.eval_batch_size=2", "train.eval_interval=1"])
    run_and_check(cfg, "top1")

    feat_dir = tmp_path / "features"
    video = sorted(glob.glob(str(tmp_path / "k400" / "videos" / "*.mp4")))[0]
    config_path = os.path.join(cfg.output_dir, "config.yaml")
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "extract_video_features.py"), "--config", config_path,
                    "--checkpoint", os.path.join(cfg.output_dir, "final.pth"),
                    "--video-dir", os.path.dirname(video), "--output-dir", str(feat_dir), "--num-frames", "4",
                    "--hop", "4", "--size", "32", "--batch-size", "2"], check=True)
    feats = torch.load(str(feat_dir / (os.path.basename(video)[:-4] + ".pt")), weights_only=True)
    assert feats.shape == (3, 32)


@pytest.mark.parametrize("dataset", ["ade20k", "cityscapes"])
def test_segmentation(imagenet_ckpt, tmp_path, dataset):
    if dataset == "ade20k":
        root = make_ade20k(str(tmp_path / "ade"))
        extra = ["data.scale=[128,32]", "data.test.scale=[128,32]", "data.test.crop_size=[32,32]",
                 "data.test.stride=[21,21]"]
    else:
        root = make_cityscapes(str(tmp_path / "cs"))
        extra = ["data.scale=[128,64]"]
    cfg = cfg_for(dataset, str(tmp_path / "out"), [
        f"data.root={root}", "data.crop_size=[32,32]", "model.out_indices=[0,1,1,1]", "model.decoder_channels=32",
        f"model.pretrained={imagenet_ckpt}", "schedule.iters=4", "schedule.warmup_iters=1", "train.batch_size=2",
        "train.eval_interval_iters=2", "train.ckpt_interval_iters=2"] + extra, backbone_prefix="model.backbone.")
    results = run_and_check(cfg, "mIoU")
    assert "Fb" in results and "aAcc" in results


def test_captioning(imagenet_ckpt, tmp_path):
    json_path = make_coco(str(tmp_path / "coco"))
    cfg = cfg_for("coco_captions", str(tmp_path / "out"), [
        f"data.karpathy_json={json_path}", f"data.image_root={tmp_path / 'coco' / 'images'}", "data.img_size=32",
        "data.min_word_count=1", "data.max_words=8", "data.test.max_words=8", "model.d_model=32", "model.depth=1",
        "model.num_heads=2", "model.max_len=10", f"model.pretrained={imagenet_ckpt}", "schedule.epochs=1",
        "schedule.warmup_epochs=0", "train.batch_size=4", "train.eval_batch_size=4"],
        backbone_prefix="model.encoder.")
    run_and_check(cfg, "CIDEr")
    with open(os.path.join(cfg.output_dir, "captions_pred.json"), encoding="utf-8") as f:
        assert len(json.load(f)) == 3


def test_grounding(tmp_path):
    files, feat_dir = make_ego4d(str(tmp_path / "ego4d"))
    cfg = load_config(os.path.join(ROOT, "configs", "ego4d_nlq.yaml"), COMMON + [
        f"data.train_json={files['train']}", f"data.val_json={files['val']}", f"data.feature_dir={feat_dir}",
        "data.feature_dim=16", "data.max_len=40", "model.d_model=32", "model.num_heads=2", "model.text_layers=1",
        "model.fusion_layers=1", "model.temporal_depth=1", "model.mixer.k=4", "schedule.epochs=2",
        "schedule.warmup_epochs=1", "schedule.kappa_warmup_epochs=1", "train.batch_size=4", "train.eval_batch_size=4",
        f"output_dir={tmp_path / 'out'}"])
    results = run_and_check(cfg, "R1@0.3")
    assert results["num_samples"] == 8


def test_lra_text(tmp_path):
    root = make_imdb(str(tmp_path / "aclImdb"))
    cfg = load_config(os.path.join(ROOT, "configs", "lra_text.yaml"), COMMON + [
        f"data.root={root}", "data.max_len=64", "model.embed_dim=32", "model.depth=1", "model.num_heads=2",
        "model.mixer.k=4", "schedule.epochs=2", "schedule.warmup_iters=1", "schedule.kappa_warmup_epochs=1",
        "train.batch_size=4", "train.eval_batch_size=4", f"output_dir={tmp_path / 'out'}"])
    results = run_and_check(cfg, "accuracy")
    assert results["num_samples"] == 4


def test_divergence_is_recorded(tmp_path):
    root = make_imdb(str(tmp_path / "aclImdb"))
    cfg = load_config(os.path.join(ROOT, "configs", "lra_text.yaml"), COMMON + [
        f"data.root={root}", "data.max_len=32", "model.embed_dim=32", "model.depth=1", "model.num_heads=2",
        "model.mixer.k=4", "schedule.epochs=3", "train.batch_size=2", "train.max_nonfinite_steps=2",
        "optim.lr=1.0e+30", "schedule.warmup_iters=0", f"output_dir={tmp_path / 'out'}"])
    results = train(cfg)
    assert results["diverged"]


def test_tools(imagenet_ckpt, tmp_path):
    out = os.path.dirname(imagenet_ckpt)
    config_path = os.path.join(out, "config.yaml")
    flops = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "count_flops.py"), "--config", config_path],
                           check=True, capture_output=True, text=True)
    assert json.loads(flops.stdout)["GFLOPs_per_image"] > 0
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "benchmark_scaling.py"), "--config", config_path,
                    "--tokens", "32", "64", "--batch-size", "2", "--dim", "32", "--heads", "2", "--repeats", "1",
                    "--warmup", "0", "--output-dir", str(tmp_path / "scaling")], check=True, capture_output=True)
    assert os.path.isfile(tmp_path / "scaling" / "scaling.png")
    subprocess.run([sys.executable, os.path.join(ROOT, "tools", "visualize_diffusion.py"), "--steps", "0", "1", "2",
                    "--output", str(tmp_path / "toy.png")], check=True, capture_output=True)
    assert os.path.isfile(tmp_path / "toy.png")
    agg = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "aggregate_results.py"),
                          os.path.join(out, "results.json")], check=True, capture_output=True, text=True)
    assert "top1" in agg.stdout
    for spec in sorted(glob.glob(os.path.join(ROOT, "configs", "ablations", "*.yaml"))):
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "run_ablation.py"), "--spec", spec, "--dry-run"],
                       check=True, capture_output=True)
