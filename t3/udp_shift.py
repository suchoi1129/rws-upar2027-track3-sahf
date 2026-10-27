"""Shift-Aligned Heatmap Fusion (SAHF) for the submission path: every box is cropped at the center view and at shifted
windows, all views go through the model in one batch (plus the flip batch), each view's heatmap is aligned to the 64x48
grid of the center view, and the aligned heatmaps are averaged, then decoded exactly as t3.udp.predict_batch_udp.
cfg["shifts"]: list of [dx, dy, unit], unit "px" = pixels of the network input, "bbox" = fraction of the cleaned box w / h.
With equal window size the alignment is a pure translation of the heatmap by -shift * (hm - 1) / window."""
import numpy as np
import torch
import torch.nn.functional as F
from t3.geometry import crop_affine, warp
from t3.udp import MEAN, STD, FLIP_INDEX, clean_box, center_scale, get_max_preds, post_dark_udp


def predict_batch_udp_shift(model, images, boxes, cfg, device):
    in_w, in_h, pad = cfg["in_w"], cfg["in_h"], cfg["padding"]
    shifts = [[0.0, 0.0, "px"]] + [list(s) for s in cfg["shifts"]]; V = len(shifts)
    crops, cs, off = [], [], []
    for img, box in zip(images, boxes):
        cb = clean_box(box, img.shape[2], img.shape[1]) or (float(box[0]), float(box[1]), max(float(box[2]), 1.0), max(float(box[3]), 1.0))
        cx, cy, sw, sh = center_scale(cb, in_w, in_h, pad)
        im = img.to(device).float(); o = []
        for dx, dy, unit in shifts:
            ox, oy = (dx * sw / (in_w - 1.0), dy * sh / (in_h - 1.0)) if unit == "px" else (dx * cb[2], dy * cb[3])
            crops.append(warp(im, crop_affine(cx + ox, cy + oy, sw, sh, in_w, in_h), in_w, in_h, "zeros")); o.append((ox / sw, oy / sh))
        cs.append((cx, cy, sw, sh)); off.append(o)
    x = (torch.stack(crops) / 255.0 - MEAN.to(device)) / STD.to(device)          # [B*V,3,H,W], views of a box adjacent
    hm = model(x).float()
    if cfg.get("flip", True):
        hm = 0.5 * (hm + model(x.flip(-1)).float().flip(-1)[:, FLIP_INDEX])
    BV, K, Hh, Wh = hm.shape; B = BV // V
    o = torch.tensor(off, dtype=torch.float32, device=hm.device).view(BV, 2)    # shift / window, per view
    v, u = torch.meshgrid(torch.arange(Hh, device=hm.device, dtype=torch.float32), torch.arange(Wh, device=hm.device, dtype=torch.float32), indexing="ij")
    us = u[None] - o[:, 0, None, None] * (Wh - 1.0); vs = v[None] - o[:, 1, None, None] * (Hh - 1.0)
    grid = torch.stack([us / (Wh - 1) * 2 - 1, vs / (Hh - 1) * 2 - 1], -1)
    hm = F.grid_sample(hm, grid, mode="bilinear", padding_mode="zeros", align_corners=True).view(B, V, K, Hh, Wh).mean(1)
    hm = hm.cpu().numpy().astype(np.float32)
    preds, maxvals = get_max_preds(hm)
    preds = post_dark_udp(preds, hm.copy(), kernel=cfg.get("udp_kernel", 11))
    cs = np.asarray(cs, np.float64)
    out = np.empty((B, K, 3), np.float32)
    out[..., 0] = preds[..., 0] * cs[:, None, 2] / (Wh - 1.0) + cs[:, None, 0] - cs[:, None, 2] * 0.5
    out[..., 1] = preds[..., 1] * cs[:, None, 3] / (Hh - 1.0) + cs[:, None, 1] - cs[:, None, 3] * 0.5
    out[..., 2] = 1.0
    mv = maxvals[..., 0]; thr = cfg.get("vis_thr", 0.2)
    valid = mv > thr
    score = np.where(valid.sum(1) > 0, (mv * valid).sum(1) / np.maximum(valid.sum(1), 1), 0.0)
    return torch.from_numpy(out), torch.from_numpy(score.astype(np.float32)), torch.from_numpy(mv)
