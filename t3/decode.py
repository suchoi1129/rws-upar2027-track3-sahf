"""Heatmap decoding in torch (used by both local eval and run.py)."""
import torch
import torch.nn.functional as F


def _gauss_kernel(ks, sigma, dev):
    r = (ks - 1) // 2
    x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2); k = k / k.sum()
    return k


def dark(hm, ks=11, sigma=2.0):
    """hm [B,K,H,W] -> coords [B,K,2] in heatmap pixels (x,y), maxvals [B,K]. DARK Taylor refinement."""
    B, K, H, W = hm.shape
    flat = hm.reshape(B, K, -1)
    maxv, idx = flat.max(-1)
    x = (idx % W).float(); y = torch.div(idx, W, rounding_mode="floor").float()
    k = _gauss_kernel(ks, sigma, hm.device)
    r = (ks - 1) // 2
    h = hm.reshape(B * K, 1, H, W)
    h = F.conv2d(F.pad(h, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    h = F.conv2d(F.pad(h, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    # rescale blurred map to keep original max (as in DARK)
    h = h.reshape(B, K, H, W)
    hmax = h.reshape(B, K, -1).max(-1).values.clamp(min=1e-10)
    h = h * (maxv.clamp(min=1e-10) / hmax)[..., None, None]
    lh = torch.log(h.clamp(min=1e-10))
    lp = F.pad(lh.reshape(B * K, 1, H, W), (1, 1, 1, 1), mode="replicate").reshape(B, K, H + 2, W + 2)
    xi = x.long() + 1; yi = y.long() + 1
    def at(dx, dy):
        return lp.gather(-1, (xi + dx).clamp(0, W + 1)[..., None, None].expand(B, K, H + 2, 1)).squeeze(-1) \
                 .gather(-1, (yi + dy).clamp(0, H + 1)[..., None]).squeeze(-1)
    c = at(0, 0)
    dx = 0.5 * (at(1, 0) - at(-1, 0)); dy = 0.5 * (at(0, 1) - at(0, -1))
    dxx = at(1, 0) - 2 * c + at(-1, 0); dyy = at(0, 1) - 2 * c + at(0, -1)
    dxy = 0.25 * (at(1, 1) - at(1, -1) - at(-1, 1) + at(-1, -1))
    det = dxx * dyy - dxy * dxy
    ok = (det.abs() > 1e-8) & (dxx < 0) & (dyy < 0)
    ox = torch.where(ok, -(dyy * dx - dxy * dy) / det, torch.zeros_like(det))
    oy = torch.where(ok, -(-dxy * dx + dxx * dy) / det, torch.zeros_like(det))
    ox, oy = ox.clamp(-1, 1), oy.clamp(-1, 1)
    return torch.stack([x + ox, y + oy], -1), maxv


def flip_heatmaps(hm, flip_index):
    return hm.flip(-1)[:, flip_index]


def to_image(coords_hm, A, in_w, in_h, hm_w, hm_h):
    """heatmap px -> crop px (UDP: (in-1)/(hm-1)) -> image px via inverse affine A [B,2,3]."""
    s = torch.tensor([(in_w - 1) / (hm_w - 1), (in_h - 1) / (hm_h - 1)], device=coords_hm.device)
    uv = coords_hm * s
    return torch.einsum("bkj,bij->bki", uv, A[:, :, :2].to(uv.device)) + A[:, None, :, 2].to(uv.device)
