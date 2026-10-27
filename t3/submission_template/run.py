"""UPAR 2027 Track 3 submission. Imports only torch, numpy, PIL, the standard library and the bundled t3 package."""
from __future__ import annotations
import json, os, sys
from typing import Any
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from t3.infer import predict_batch  # noqa: E402
from t3.udp import predict_batch_udp  # noqa: E402
from t3.udp_shift import predict_batch_udp_shift  # noqa: E402

_MODEL = None; _DEV = None; _CFG: dict = {}; _PRIOR = None


class _Half(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m(x.half())


def load_model() -> None:
    global _MODEL, _DEV, _CFG, _PRIOR
    if _MODEL is not None:
        return
    _CFG = json.load(open(os.path.join(HERE, "assets", "config.json")))
    _PRIOR = np.asarray(_CFG["prior_pose"], np.float32)          # [17,2] box-normalized mean pose
    _DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    use_half = bool(_CFG.get("half")) and _DEV.type == "cuda"
    _CFG["half"] = use_half
    m = torch.jit.load(os.path.join(HERE, "assets", "model.torchscript"), map_location=_DEV).eval()
    _MODEL = _Half(m.half()) if use_half else m.float()
    # warm-up so the first real images are not slowed by TorchScript profiling
    x = torch.zeros(2, 3, _CFG["in_h"], _CFG["in_w"], device=_DEV)
    for _ in range(3):
        _MODEL(x); _MODEL(x.flip(-1))
    print(f"[submission] model ready on {_DEV} ({_CFG['in_w']}x{_CFG['in_h']}, half={use_half})", flush=True)


def _prior(box):
    x, y, w, h = [float(v) for v in box[:4]]
    return [[x + u * w, y + v * h, 1.0] for u, v in _PRIOR]


def predict_image(sample: dict[str, Any]) -> dict[str, Any]:
    if _MODEL is None:
        load_model()
    boxes = sample.get("boxes") or []
    if not boxes:
        return {"keypoints": [], "scores": []}
    img = torch.from_numpy(np.array(Image.open(sample["image_path"]).convert("RGB"))).permute(2, 0, 1).contiguous()
    kps: list = [None] * len(boxes); scores: list = [0.0] * len(boxes)
    ok = [i for i, b in enumerate(boxes) if len(b) >= 4 and float(b[2]) > 0 and float(b[3]) > 0]
    for i in range(len(boxes)):
        if i not in ok:
            kps[i] = _prior(boxes[i]) if len(boxes[i]) >= 4 else [[0.0, 0.0, 1.0]] * 17
            scores[i] = 1e-3 * (1 + i * 1e-3)
    bs = int(_CFG.get("batch", 32))
    for s in range(0, len(ok), bs):
        idx = ok[s:s + bs]
        fn = (predict_batch_udp_shift if _CFG.get("shifts") else predict_batch_udp) if _CFG.get("decode") == "udp" else predict_batch
        k, sc, _ = fn(_MODEL, [img] * len(idx), [boxes[i] for i in idx], _CFG, _DEV)
        k = torch.nan_to_num(k.float(), nan=0.0).numpy(); sc = torch.nan_to_num(sc.float(), nan=0.0).numpy()
        for j, i in enumerate(idx):
            kps[i] = [[float(x), float(y), float(v)] for x, y, v in k[j]]
            scores[i] = float(sc[j])
    return {"keypoints": kps, "scores": scores}
