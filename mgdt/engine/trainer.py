"""Training and evaluation loops shared by all benchmarks."""

import contextlib
import json
import os
import time

import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel

from ..layers import mean_energy_reduction, set_energy_tracking, set_schedule_progress
from ..ops.coarsening import set_tie_break_seed
from ..utils.misc import count_parameters, save_json, seed_everything, setup_logger
from .checkpoint import load_checkpoint, model_weights, save_checkpoint, strip_prefix
from .distributed import all_reduce_min, barrier, cleanup, get_device, init_distributed
from .ema import ModelEMA
from .loader import steps_per_epoch
from .optim import LRSchedule, build_optimizer
from .tasks import build_task, move_to


def _grad_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _setup(cfg):
    info = init_distributed()
    device = get_device(info)
    seed_everything(int(cfg.seed), bool(cfg.get("deterministic", False)))
    logger = setup_logger(cfg.output_dir, info.rank)
    if info.is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    return info, device, logger


def _build_model(cfg, task, info, device, logger):
    model = task.build_model()
    init = cfg.model.get("pretrained")
    if init:
        state = strip_prefix(model_weights(load_checkpoint(init), bool(cfg.model.get("pretrained_ema", True))))
        missing, unexpected = task.init_from_pretrained(model, state)
        logger.info("Initialised from %s (%d missing, %d unexpected keys)", init, len(missing), len(unexpected))
        if missing:
            logger.info("Missing keys: %s", ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))
        if unexpected:
            logger.info("Unexpected keys: %s", ", ".join(unexpected[:20]) + (" ..." if len(unexpected) > 20 else ""))
    model.to(device)
    if info.enabled and task.sync_batchnorm:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    logger.info("Model parameters: %.2fM", count_parameters(model) / 1e6)
    return model


def _schedule_lengths(cfg, spe):
    sch = cfg.schedule
    total = int(sch.epochs) * spe if sch.get("epochs") is not None else int(sch.iters)
    if sch.get("warmup_iters") is not None:
        warmup = int(sch.warmup_iters)
    else:
        warmup = int(round(float(sch.get("warmup_epochs", 0)) * spe))
    if sch.get("kappa_warmup_epochs") is not None:
        kappa = int(round(float(sch.kappa_warmup_epochs) * spe))
    elif sch.get("kappa_warmup_frac") is not None:
        kappa = int(round(float(sch.kappa_warmup_frac) * total))
    else:
        kappa = 0
    return total, warmup, kappa


def _evaluate_all(task, model, ema, loader, logger, track_energy=False):
    if track_energy:
        set_energy_tracking(model, True)
    metrics = task.evaluate(model, loader)
    if track_energy:
        reduction = mean_energy_reduction(model)
        if reduction is not None:
            metrics["energy_reduction"] = 100.0 * reduction
        set_energy_tracking(model, False)
    if ema is not None:
        for key, value in task.evaluate(ema.module, loader).items():
            metrics[f"ema_{key}"] = value
    logger.info("Evaluation: %s", json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                              for k, v in metrics.items()}))
    return metrics


def _primary(task, metrics):
    keys = [task.primary_metric, f"ema_{task.primary_metric}"]
    values = [metrics[k] for k in keys if k in metrics]
    return max(values) if task.higher_is_better else min(values)


def train(cfg):
    """Trains ``cfg.task`` and writes checkpoints, logs and ``results.json`` to ``cfg.output_dir``."""
    info, device, logger = _setup(cfg)
    task = build_task(cfg, device, info, logger)
    model = _build_model(cfg, task, info, device, logger)
    seed_everything(int(cfg.seed) + info.rank, bool(cfg.get("deterministic", False)))
    set_tie_break_seed(int(cfg.seed))

    train_loader, train_sampler = task.build_train_loader()
    eval_loader = task.build_eval_loader()
    tcfg = cfg.train
    accum = max(1, int(tcfg.get("accum_steps", 1)))
    spe = steps_per_epoch(train_loader, accum)
    total, warmup, kappa_steps = _schedule_lengths(cfg, spe)
    sch = cfg.schedule
    lr_at = LRSchedule(cfg.optim.lr, total, warmup, sch.get("scheduler", "cosine"), sch.get("min_lr", 0.0),
                       sch.get("power", 1.0))
    optimizer = build_optimizer(model, cfg.optim)
    amp = tcfg.get("amp", "none")
    scaler = _grad_scaler(amp == "fp16" and device.type == "cuda")
    ema_cfg = tcfg.get("ema") or {}
    ema = ModelEMA(model, float(ema_cfg.get("decay", 0.9999))) if ema_cfg.get("enabled", False) else None
    net = model
    if info.enabled:
        net = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                      find_unused_parameters=bool(tcfg.get("find_unused_parameters", False)))

    step, epoch, best = 0, 0, None
    last_path = os.path.join(cfg.output_dir, "last.pth")
    resume = tcfg.get("resume") or (last_path if tcfg.get("auto_resume", True) and os.path.isfile(last_path) else None)
    if resume:
        ckpt = load_checkpoint(resume)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if ema is not None and ckpt.get("ema"):
            ema.load_state_dict(ckpt["ema"])
        step, epoch = int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))
        best = ckpt.get("metrics", {}).get("best")
        logger.info("Resumed from %s at step %d (epoch %d)", resume, step, epoch)

    eval_every_epochs = tcfg.get("eval_interval")
    eval_every_steps = tcfg.get("eval_interval_iters")
    ckpt_every_steps = tcfg.get("ckpt_interval_iters")
    log_every = int(tcfg.get("log_interval", 50))
    clip_grad = tcfg.get("clip_grad")
    max_nonfinite = int(tcfg.get("max_nonfinite_steps", 20))
    metrics_path = os.path.join(cfg.output_dir, "metrics.jsonl")
    logger.info("Training for %d optimiser steps (%d per epoch), warm-up %d, kappa warm-up %d",
                total, spe, warmup, kappa_steps)

    track_energy = bool(tcfg.get("track_energy_at_eval", False))

    def evaluate_and_save(tag):
        nonlocal best
        metrics = _evaluate_all(task, model, ema, eval_loader, logger, track_energy)
        metrics.update({"step": step, "epoch": epoch, "tag": tag})
        score = _primary(task, metrics)
        improved = best is None or (score > best if task.higher_is_better else score < best)
        if improved:
            best = score
        if info.is_main:
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
            if improved:
                save_checkpoint(os.path.join(cfg.output_dir, "best.pth"), model, ema=ema, step=step, epoch=epoch,
                                cfg=cfg, metrics={"best": best})
        barrier()
        model.train()
        return metrics

    def save_last():
        if info.is_main:
            save_checkpoint(last_path, model, optimizer, scaler, ema, step, epoch, cfg, {"best": best})
        barrier()

    diverged = False
    nonfinite = 0
    loss_sum, loss_count = 0.0, 0
    first_step = step
    train_time = 0.0
    tic = time.time()
    step_tic = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    while step < total and not diverged:
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        skip_window = False
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(train_loader):
            if step >= total:
                break
            if micro // accum >= spe:
                break
            last_micro = (micro + 1) % accum == 0
            progress = step / kappa_steps if kappa_steps > 0 else 1.0
            set_schedule_progress(model, progress)
            if ema is not None:
                set_schedule_progress(ema.module, progress)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            batch = move_to(batch, device)
            sync = net.no_sync() if (info.enabled and not last_micro) else contextlib.nullcontext()
            with sync:
                with task.autocast():
                    loss, logs = task.train_step(net, batch)
                finite = torch.tensor([1.0 if bool(torch.isfinite(loss.detach())) else 0.0], device=device)
                all_reduce_min(finite)
                if finite.item() < 1.0:
                    skip_window = True
                else:
                    scaler.scale(loss / accum).backward()
                    loss_sum += float(loss.detach())
                    loss_count += 1
            if not last_micro:
                continue
            if skip_window:
                optimizer.zero_grad(set_to_none=True)
                skip_window = False
                nonfinite += 1
                logger.warning("Non-finite loss at step %d (%d consecutive)", step, nonfinite)
                if nonfinite >= max_nonfinite:
                    diverged = True
                    break
                continue
            nonfinite = 0
            if clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            step += 1
            train_time += time.time() - step_tic
            if step % log_every == 0 or step == total:
                elapsed = time.time() - tic
                eta = elapsed / log_every * (total - step) if step % log_every == 0 else 0.0
                mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
                extra = " ".join(f"{k} {v:.4f}" for k, v in logs.items())
                logger.info("epoch %d step %d/%d loss %.4f lr %.3e %s| %.2fs/step eta %.1fh mem %.1fG",
                            epoch, step, total, loss_sum / max(1, loss_count), optimizer.param_groups[0]["lr"],
                            extra + " " if extra else "", elapsed / log_every, eta / 3600, mem)
                loss_sum, loss_count = 0.0, 0
                tic = time.time()
            if eval_every_steps and step % int(eval_every_steps) == 0 and step < total:
                evaluate_and_save(f"step{step}")
            if ckpt_every_steps and step % int(ckpt_every_steps) == 0:
                save_last()
            step_tic = time.time()
        epoch += 1
        if diverged or step >= total:
            break
        if eval_every_epochs and epoch % int(eval_every_epochs) == 0:
            evaluate_and_save(f"epoch{epoch}")
        save_last()
        step_tic = time.time()

    results = {"diverged": diverged, "step": step, "epoch": epoch,
               "avg_step_time_s": train_time / max(1, step - first_step)}
    if device.type == "cuda":
        results["peak_memory_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
    if diverged:
        logger.error("Training diverged after %d consecutive non-finite steps", max_nonfinite)
    else:
        final = evaluate_and_save("final")
        results.update({k: v for k, v in final.items() if k not in ("tag",)})
        results["best"] = best
        save_last()
        if info.is_main:
            save_checkpoint(os.path.join(cfg.output_dir, "final.pth"), model, ema=ema, step=step, epoch=epoch,
                            cfg=cfg, metrics={"best": best})
    if info.is_main:
        save_json(results, os.path.join(cfg.output_dir, "results.json"))
    barrier()
    cleanup()
    return results


def evaluate(cfg, checkpoint, use_ema=False, track_energy=False):
    """Evaluates a checkpoint with the task's evaluation protocol and writes ``eval_results.json``."""
    info, device, logger = _setup(cfg)
    task = build_task(cfg, device, info, logger)
    model = task.build_model()
    state = strip_prefix(model_weights(load_checkpoint(checkpoint), prefer_ema=use_ema))
    model.load_state_dict(state)
    model.to(device)
    set_schedule_progress(model, 1.0)
    metrics = _evaluate_all(task, model, None, task.build_eval_loader(), logger, track_energy)
    metrics["checkpoint"] = os.path.abspath(checkpoint)
    metrics["weights"] = "ema" if use_ema else "model"
    if info.is_main:
        save_json(metrics, os.path.join(cfg.output_dir, "eval_results.json"))
    barrier()
    cleanup()
    return metrics
