import pytest
import torch

from mgdt.layers import Block, build_mixer
from mgdt.models import (MGDTCaptioner, MGDTGroundingHead, MGDTSegmenter, MGDTTextClassifier, MGDTVideoTransformer,
                         MGDTVisionTransformer)

MIXER = {"k": 6, "levels": 1}
TINY = {"img_size": 32, "patch_size": 8, "embed_dim": 32, "depth": 2, "num_heads": 2, "mixer": MIXER}


@pytest.mark.parametrize("cfg", [
    {}, {"levels": 0}, {"levels": 2, "coarsen_factor": 4}, {"coarsening": "kmeans"}, {"conductance": "rational"},
    {"conductance": "isotropic"}, {"threshold": 2.0}, {"kappa_mode": "layer"}, {"similarity": "query"},
    {"similarity": "shared"}, {"similarity": "qk"}, {"similarity": "asym"}, {"metric": "cos"},
    {"normalization": "none"}, {"type": "softmax"}, {"type": "softmax", "shared_qk": True},
])
def test_mixer_variants_forward_backward(cfg):
    torch.manual_seed(0)
    m = build_mixer(32, 4, dict(cfg, k=6))
    x = torch.randn(2, 36, 32, requires_grad=True)
    out = m(x, (6, 6))
    out.sum().backward()
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_padding_does_not_leak():
    torch.manual_seed(0)
    m = build_mixer(32, 4, {"k": 6, "levels": 2}).eval()
    x = torch.randn(2, 40, 32)
    mask = torch.ones(2, 40, dtype=torch.bool)
    mask[1, 30:] = False
    x2 = x.clone()
    x2[1, 30:] = 100 * torch.randn(10, 32)
    assert torch.allclose(m(x, (40,), mask)[1, :30], m(x2, (40,), mask)[1, :30], atol=1e-5)


def test_mixer_parameter_parity_with_shared_qk_softmax():
    a = sum(p.numel() for p in build_mixer(64, 4, {"type": "mgdt", "kappa_mode": "fixed"}).parameters())
    b = sum(p.numel() for p in build_mixer(64, 4, {"type": "softmax", "shared_qk": True}).parameters())
    assert a == b


def test_block_and_backbone():
    vit = MGDTVisionTransformer(num_classes=5, **TINY)
    assert vit(torch.randn(2, 3, 32, 32)).shape == (2, 5)
    assert vit(torch.randn(1, 3, 48, 40)).shape == (1, 5)
    blk = Block(32, 2, mixer=MIXER)
    assert blk(torch.randn(1, 64, 32), (4, 4, 4)).shape == (1, 64, 32)


def test_video_model_and_inflation():
    vit = MGDTVisionTransformer(num_classes=5, **TINY)
    video = MGDTVideoTransformer(num_classes=7, num_frames=4, tubelet_size=2, temporal_depth=1, **TINY)
    missing, unexpected = video.load_image_state_dict(vit.state_dict())
    assert "patch_embed.weight" not in missing and not unexpected
    w2 = vit.patch_embed.weight
    assert torch.allclose(video.patch_embed.weight.sum(2), w2, atol=1e-6)
    out = video(torch.randn(2, 3, 4, 32, 32))
    assert out.shape == (2, 7)
    out.sum().backward()


def test_segmenter_shapes():
    seg = MGDTSegmenter(11, backbone=TINY, out_indices=(0, 1, 1, 1), decoder_channels=32)
    out = seg(torch.randn(2, 3, 50, 70))
    assert out.shape == (2, 11, 50, 70)


def test_captioner_beam_search():
    torch.manual_seed(0)
    cap = MGDTCaptioner(30, encoder=TINY, d_model=32, depth=2, num_heads=2, max_len=12).eval()
    tokens = torch.randint(4, 30, (2, 8))
    assert cap(torch.randn(2, 3, 32, 32), tokens).shape == (2, 8, 30)
    out = cap.generate(torch.randn(3, 3, 32, 32), beam_size=3, max_words=8)
    assert len(out) == 3 and all(len(o) <= 8 for o in out)
    assert all(t not in (0, 1, 2, 3) for o in out for t in o)


def test_incremental_decoding_matches_full_decoding():
    torch.manual_seed(0)
    cap = MGDTCaptioner(30, encoder=TINY, d_model=32, depth=2, num_heads=4, max_len=12).eval()
    images = torch.randn(2, 3, 32, 32)
    tokens = torch.randint(4, 30, (2, 9))
    tokens[:, 0] = cap.bos_id
    with torch.no_grad():
        full = cap(images, tokens)
        memory_kv, caches = cap.start_decoding(images, 1)
        steps = [cap.decode_step(tokens[:, t], t, memory_kv, caches) for t in range(tokens.shape[1])]
    assert torch.allclose(full, torch.stack(steps, 1), atol=1e-5)


def test_grounding_and_text_models():
    g = MGDTGroundingHead(50, video_dim=16, d_model=32, num_heads=2, temporal_depth=1, mixer=MIXER)
    video = torch.randn(2, 20, 16)
    vmask = torch.ones(2, 20, dtype=torch.bool)
    vmask[1, 15:] = False
    words = torch.randint(1, 50, (2, 5))
    s, e, i = g(video, vmask, words, torch.ones(2, 5, dtype=torch.bool))
    assert s.shape == (2, 20) and (s[1, 15:] <= -1e3).all()
    txt = MGDTTextClassifier(embed_dim=32, depth=1, num_heads=2, mixer=MIXER)
    tokens = torch.randint(1, 257, (2, 64))
    tokens[0, 40:] = 0
    assert txt(tokens).shape == (2, 2)
