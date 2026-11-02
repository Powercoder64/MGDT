"""COCO Captions (Karpathy split) with CIDEr-D evaluation."""

import json
import os

import torch
import torch.nn.functional as F

from ...data import (CocoCaptionEval, CocoCaptionTrain, build_caption_vocab, caption_eval_transform,
                     caption_train_transform, load_karpathy)
from ...metrics import CiderD
from ..checkpoint import unwrap
from ..distributed import all_gather_objects, broadcast_object
from ..loader import make_loader, per_process_batch
from .base import Task


class CaptioningTask(Task):
    primary_metric = "CIDEr"

    def __init__(self, cfg, device, dist_info, logger):
        super().__init__(cfg, device, dist_info, logger)
        self.records = load_karpathy(cfg.data.karpathy_json)
        self.vocab = build_caption_vocab(self.records, int(cfg.data.get("min_word_count", 5)))
        if dist_info.is_main and cfg.get("output_dir"):
            os.makedirs(cfg.output_dir, exist_ok=True)
            self.vocab.save(os.path.join(cfg.output_dir, "vocab.json"))
        logger.info("Caption vocabulary: %d words", len(self.vocab))

    def model_kwargs(self):
        v = self.vocab
        return {"vocab_size": len(v), "pad_id": v.pad_id, "bos_id": v.bos_id, "eos_id": v.eos_id, "unk_id": v.unk_id}

    def init_from_pretrained(self, model, state_dict):
        state = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
        result = unwrap(model).encoder.load_state_dict(state, strict=False)
        return list(result.missing_keys), list(result.unexpected_keys)

    def build_train_loader(self):
        d = self.cfg.data
        dataset = CocoCaptionTrain(d.image_root, self.records, self.vocab,
                                   caption_train_transform(int(d.img_size), tuple(d.get("crop_scale", (0.5, 1.0)))),
                                   max_words=int(d.max_words))
        bs = per_process_batch(self.cfg.train.batch_size, self.dist, self.cfg.train.get("accum_steps", 1))
        return make_loader(dataset, bs, True, self.dist, self.num_workers, seed=int(self.cfg.seed))

    def build_eval_loader(self):
        d = self.cfg.data
        self.eval_dataset = CocoCaptionEval(d.image_root, self.records, d.get("eval_split", "test"),
                                            caption_eval_transform(int(d.img_size)))
        loader, _ = make_loader(self.eval_dataset, self.eval_batch_size(), False, self.dist, self.num_workers)
        return loader

    def train_step(self, model, batch):
        images, seq = batch
        logits = model(images, seq[:, :-1])
        target = seq[:, 1:]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1),
                               ignore_index=self.vocab.pad_id)
        return loss, {}

    @torch.no_grad()
    def evaluate(self, model, loader):
        model.eval()
        net = unwrap(model)
        test = self.cfg.data.get("test", {})
        beam, max_words = int(test.get("beam_size", 3)), int(test.get("max_words", 20))
        local = []
        for images, index in loader:
            images = images.to(self.device, non_blocking=True)
            with self.autocast():
                outputs = net.generate(images, beam_size=beam, max_words=max_words)
            for i, words in zip(index.tolist(), outputs):
                local.append((i, " ".join(self.vocab.decode(words))))
        gathered = [item for part in all_gather_objects(local) for item in part]
        metrics = None
        if self.dist.is_main:
            hyps = {i: c for i, c in gathered}
            refs = {i: self.eval_dataset.references[i] for i in hyps}
            score, _ = CiderD().compute_score(refs, hyps)
            metrics = {"CIDEr": 100.0 * score, "num_samples": len(hyps)}
            if self.cfg.get("output_dir"):
                records = self.eval_dataset.records
                preds = [{"image_id": records[i].get("cocoid", i), "caption": hyps[i]} for i in sorted(hyps)]
                with open(os.path.join(self.cfg.output_dir, "captions_pred.json"), "w", encoding="utf-8") as f:
                    json.dump(preds, f)
        return broadcast_object(metrics)
