"""Batched top-down inference shared by local evaluation and the submission (torch only)."""
import torch
from t3.common import FLIP_INDEX, EVAL_SLOTS
from t3.geometry import box_center_scale, crop_affine, warp, mask_outside_box
from t3.decode import dark, flip_heatmaps, to_image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def predict_batch(model, images, boxes, cfg, device):
    """images: list of uint8 tensors [3,H,W] (one per box); boxes: list [x,y,w,h].
    cfg: in_w, in_h, padding, flip, dark_ks, dark_sigma, pad_mode, tta_pads (optional list of paddings).
    Multi-padding TTA: every padding is a separate crop in ONE batch; per-joint image coordinates are averaged
    with heatmap-max weights. Returns kps [B,17,3] (image px), scores [B], maxv [B,17]."""
    in_w, in_h = cfg["in_w"], cfg["in_h"]
    pads = cfg.get("tta_pads") or [cfg["padding"]]
    B, S = len(images), len(pads)
    crops, As = [], []
    for p in pads:
        for img, box in zip(images, boxes):
            cx, cy, sw, sh = box_center_scale(box, in_w, in_h, p)
            A = crop_affine(cx, cy, sw, sh, in_w, in_h)
            src = mask_outside_box(img.to(device).float(), box, cfg.get("box_mask", 0.0))
            crops.append(warp(src, A, in_w, in_h, cfg.get("pad_mode", "zeros")))
            As.append(A)
    x = torch.stack(crops) / 255.0
    x = (x - MEAN.to(device)) / STD.to(device)
    A = torch.stack(As).to(device)
    hm = model(x).float()
    ks, sg = cfg.get("dark_ks", 11), cfg.get("dark_sigma", 2.0)
    dis = None
    if cfg.get("flip", True):
        hf = flip_heatmaps(model(x.flip(-1)).float(), FLIP_INDEX)
        if cfg.get("extra"):
            # flip consistency (heatmap px) between original and flipped-back predictions (analysis only)
            dis = (dark(hm, ks, sg)[0] - dark(hf, ks, sg)[0]).norm(dim=-1)
        hm = 0.5 * (hm + hf)
    hm_h, hm_w = hm.shape[-2:]
    xy, maxv = dark(hm, ks=ks, sigma=sg)
    pts = to_image(xy, A, in_w, in_h, hm_w, hm_h).view(S, B, -1, 2)
    maxv = maxv.view(S, B, -1)
    w = maxv.clamp(min=1e-6)[..., None]
    pts = (pts * w).sum(0) / w.sum(0)
    score = maxv[:, :, EVAL_SLOTS].mean(2).mean(0)
    maxv = maxv.mean(0)
    kps = torch.cat([pts, torch.ones_like(maxv)[..., None]], -1)
    if cfg.get("extra"):
        return kps.cpu(), score.cpu(), maxv.cpu(), (dis if dis is not None else torch.zeros_like(maxv)).cpu()
    return kps.cpu(), score.cpu(), maxv.cpu()
