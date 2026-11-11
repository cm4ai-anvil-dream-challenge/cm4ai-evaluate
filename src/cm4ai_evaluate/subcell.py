"""SubCell features for the panel metrics (needs the [subcell] extra: torch).

SubCell is a vision transformer trained on Human Protein Atlas images
(Gupta et al., "SubCell: Vision foundation models for microscopy capture
single-cell biology", bioRxiv 2024, https://doi.org/10.1101/2024.12.06.627299).
The weights are the public rybg MAE-CellS-ProtS-Pool encoder, downloaded once
(349 MB) and checked against a fixed sha256.

The model code and the input preparation are copied from dream_eval 0.3.0
(cm4ai-anvil-dream-tools, branch refactor/shared-organizer-manifest-subsets,
commit 581edc9: dream_eval/subcell_vit.py and Extractor in
dream_eval/metrics_math.py), which in turn copies the evaluation team's spec
modules c3_subcell_vit.py and c3metrics.py. Changes: names, comments, and
the weights download; the computation is unchanged. Precision: fp32 by
default; the challenge grading runs fp16 on CUDA (dream_eval reports feature
cosine of at least 0.9999 against fp32).

Two inputs per image ("arms"), both multiplied by the cell mask:
  protein:  the protein channel alone, min-max scaled to [0, 1]
  full:     microtubules, ER, nucleus, protein, all scaled by the min and max
            of the three reference channels (so protein brightness relative to
            the references is kept)
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .io import ImageCheckError

WEIGHTS_URL = "https://czi-subcell-public.s3.amazonaws.com/models/all_channels_MAE-CellS-ProtS-Pool.pth"
WEIGHTS_SHA256 = "493ff6d17e108e89b805f79a17cf09c2f237458637ed847e3ae7d187295f387d"
IMAGE_SIZE = 512
PROTEIN = "protein"
FULL = "full"
ARMS = (PROTEIN, FULL)


def default_weights_path() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "cm4ai-evaluate" / "subcell" / "all_channels_MAE-CellS-ProtS-Pool.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_weights(path: str | Path | None = None, say=print) -> Path:
    """Return a checked weights file, downloading it to the cache if no path is given."""
    if path is not None:
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SubCell weights not found: {path}")
    else:
        path = default_weights_path()
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            say(f"Downloading SubCell weights (349 MB) to {path} ...")
            partial = path.with_suffix(".part")
            urllib.request.urlretrieve(WEIGHTS_URL, partial)
            partial.replace(path)
    if sha256(path) != WEIGHTS_SHA256:
        raise ValueError(f"{path}: sha256 does not match the public SubCell weights; delete it and retry")
    return path


# Model: copied from dream_eval/subcell_vit.py (see the module docstring).


class _Block(nn.Module):
    def __init__(self, d=768, h=12, mlp=3072, eps=1e-12):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d, eps=eps)
        self.ln2 = nn.LayerNorm(d, eps=eps)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, mlp)
        self.fc2 = nn.Linear(mlp, d)

    def forward(self, x):
        b, n, d = x.shape
        y = self.ln1(x)
        q, k, v = (t(y).view(b, n, self.h, d // self.h).transpose(1, 2) for t in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o(a.transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class SubCellViT(nn.Module):
    def __init__(self, in_ch=4, d=768, depth=12, heads=12, patch=16, img=448, pool_heads=2, pool_int=512):
        super().__init__()
        self.patch = patch
        self.grid0 = img // patch
        self.proj = nn.Conv2d(in_ch, d, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, self.grid0**2 + 1, d))
        self.blocks = nn.ModuleList([_Block(d, heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(d, eps=1e-12)
        self.pv = nn.Linear(d, pool_int)
        self.pu = nn.Linear(d, pool_int)
        self.pa = nn.Linear(pool_int, pool_heads)
        self._pos_cache = {}

    def _pos(self, gh, gw):
        if gh == self.grid0 and gw == self.grid0:
            return self.pos
        # Interpolated position grid, cached: fixed for given weights and size
        # (dream_eval caches it the same way; the values are bit-identical).
        key = (gh, gw, self.pos.dtype, self.pos.device, torch.is_autocast_enabled())
        if key not in self._pos_cache:
            cp, pp = self.pos[:, :1], self.pos[:, 1:]
            d = pp.shape[-1]
            pp = pp.reshape(1, self.grid0, self.grid0, d).permute(0, 3, 1, 2)
            pp = F.interpolate(pp, size=(gh, gw), mode="bicubic", align_corners=False)
            self._pos_cache.clear()
            self._pos_cache[key] = torch.cat([cp, pp.permute(0, 2, 3, 1).reshape(1, -1, d)], 1)
        return self._pos_cache[key]

    def forward(self, x):
        b, c, H, W = x.shape
        t = self.proj(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(b, -1, -1), t], 1) + self._pos(H // self.patch, W // self.patch)
        for block in self.blocks:
            t = block(t)
        t = self.ln(t)
        attn = self.pa(torch.tanh(self.pv(t)) * F.gelu(self.pu(t))).permute(0, 2, 1).softmax(-1)
        return torch.bmm(attn, t).reshape(b, -1)  # pooled features, 2 heads x 768 = 1536


def load_model(weights: Path, device) -> SubCellViT:
    model = SubCellViT()
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    sd = {
        "proj.weight": ck["encoder.embeddings.patch_embeddings.projection.weight"],
        "proj.bias": ck["encoder.embeddings.patch_embeddings.projection.bias"],
        "cls": ck["encoder.embeddings.cls_token"],
        "pos": ck["encoder.embeddings.position_embeddings"],
        "ln.weight": ck["encoder.layernorm.weight"],
        "ln.bias": ck["encoder.layernorm.bias"],
    }
    names = {
        "ln1": "layernorm_before", "ln2": "layernorm_after", "q": "attention.attention.query",
        "k": "attention.attention.key", "v": "attention.attention.value", "o": "attention.output.dense",
        "fc1": "intermediate.dense", "fc2": "output.dense",
    }
    for i in range(12):
        prefix = f"encoder.encoder.layer.{i}."
        for ours, theirs in names.items():
            for w in ("weight", "bias"):
                sd[f"blocks.{i}.{ours}.{w}"] = ck[prefix + theirs + "." + w]
    for ours, theirs in (("pv", "attention_v.1"), ("pu", "attention_u.1"), ("pa", "attention")):
        for w in ("weight", "bias"):
            sd[f"{ours}.{w}"] = ck[f"pool_model.{theirs}.{w}"]
    model.load_state_dict(sd, strict=True)
    n_encoder = sum(1 for k in ck if k.startswith("encoder.") or k.startswith("pool_model."))
    if n_encoder != len(sd):
        raise ValueError(f"unexpected SubCell checkpoint layout ({n_encoder} encoder tensors, {len(sd)} mapped)")
    return model.eval().to(device)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SubCellExtractor:
    """Loads the model once; features(protein, references, mask, arms) for a batch of cells."""

    def __init__(self, weights: str | Path | None = None, device: str | None = None, batch: int = 8, say=print):
        self.device = torch.device(device or default_device())
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.weights = ensure_weights(weights, say=say)
        self.model = load_model(self.weights, self.device)
        self.batch = batch

    def _input(self, p, r, m, arm):
        # Copied from dream_eval Extractor.subcell_input (channel order microtubules, ER, nucleus, protein).
        p = p * m
        x = torch.zeros((p.shape[0], 4) + p.shape[1:], device=self.device)
        x[:, 3] = p
        if arm == FULL:
            r = r * m[:, None]
            x[:, 0] = r[:, 1]
            x[:, 1] = r[:, 2]
            x[:, 2] = r[:, 0]
            src = x[:, :3]
            mn = src.amin(dim=(1, 2, 3), keepdim=True)
            mx = src.amax(dim=(1, 2, 3), keepdim=True)
            degenerate = (mx - mn) <= 1e-8
            x = ((x - mn) / (mx - mn + 1e-8)).clamp(0, 1)
            x = torch.where(degenerate, torch.zeros_like(x), x)
        elif arm == PROTEIN:
            mn = x.amin(dim=(1, 2, 3), keepdim=True)
            mx = x.amax(dim=(1, 2, 3), keepdim=True)
            x = (x - mn) / (mx - mn + 1e-8)
        else:
            raise ValueError(arm)
        return x

    def features(self, protein, mask, references=None, arms=(PROTEIN,)) -> dict[str, np.ndarray]:
        """protein [n, 512, 512] float32 in [0, 1]; references [n, 3, 512, 512] as nucleus,
        microtubules, ER (needed for the full arm); mask [n, 512, 512] bool.
        Returns {arm: [n, 1536] float32}; rows with an empty mask are nan."""
        protein = np.asarray(protein, np.float32)
        mask = np.asarray(mask, bool)
        if protein.ndim != 3 or mask.shape != protein.shape:
            raise ImageCheckError("SubCell", "shape", "protein and mask must have matching [n, height, width] shapes")
        patch = self.model.patch
        if min(protein.shape[1:]) < patch:
            raise ImageCheckError("SubCell", "size", f"{protein.shape[1]} x {protein.shape[2]} image needs at least {patch} pixels per dimension")
        if references is not None and np.shape(references) != (len(protein), 3, *protein.shape[1:]):
            raise ImageCheckError("SubCell", "shape", "reference channels must have shape [n, 3, height, width]")
        if FULL in arms and references is None:
            raise ValueError("the full arm needs the reference channels")
        out = {a: [] for a in arms}
        with torch.no_grad():
            for s in range(0, len(protein), self.batch):
                p = torch.from_numpy(protein[s : s + self.batch]).to(self.device, torch.float32)
                m = torch.from_numpy(mask[s : s + self.batch]).to(self.device).to(torch.float32)
                r = None
                if references is not None:
                    r = torch.from_numpy(np.asarray(references[s : s + self.batch], np.float32)).to(self.device)
                for a in arms:
                    out[a].append(self.model(self._input(p, r, m, a)).float().cpu().numpy())
        result = {a: np.concatenate(v).astype(np.float32) for a, v in out.items()}
        empty = ~mask.reshape(len(mask), -1).any(1)
        for a in arms:
            result[a][empty] = np.nan
        return result


def load_location_head(path: str | Path) -> dict:
    """A linear location head: npz with W [classes, 1536], b [classes], classes [classes]."""
    z = np.load(Path(path).expanduser(), allow_pickle=False)
    head = {"W": np.asarray(z["W"], np.float64), "b": np.asarray(z["b"], np.float64)}
    head["classes"] = [str(c) for c in z["classes"]] if "classes" in z else []
    if head["W"].ndim != 2 or head["W"].shape[1] != 1536 or head["b"].shape != (head["W"].shape[0],):
        raise ValueError(f"{path}: expected W [classes, 1536] and b [classes]")
    return head


__all__ = ["SubCellExtractor", "ensure_weights", "load_location_head", "ARMS", "PROTEIN", "FULL"]
