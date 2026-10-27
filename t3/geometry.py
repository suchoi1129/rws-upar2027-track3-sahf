"""Crop geometry shared by training and the submission (torch only, UDP convention: pixel centers,
size-1 spans). An affine A (2x3) maps OUTPUT crop pixel coords -> INPUT image pixel coords."""
import math
import torch


def box_center_scale(box, out_w, out_h, padding):
    x, y, w, h = [float(v) for v in box[:4]]
    w, h = max(w, 1.0), max(h, 1.0)
    cx, cy = x + 0.5 * w, y + 0.5 * h
    aspect = out_w / out_h
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    return cx, cy, w * padding, h * padding          # crop window size in input pixels


def crop_affine(cx, cy, sw, sh, out_w, out_h, rot_deg=0.0, flip=False):
    """Inverse affine: output (u,v) -> input (x,y). The window (cx,cy,sw,sh) is mapped onto [0,out_w-1]x[0,out_h-1]."""
    sx, sy = sw / (out_w - 1.0), sh / (out_h - 1.0)
    r = math.radians(rot_deg); c, s = math.cos(r), math.sin(r)
    fx = -1.0 if flip else 1.0
    # centered output coords: u' = u - (out_w-1)/2 ; input = center + R @ diag(fx*sx, sy) @ [u', v']
    a00, a01 = c * fx * sx, -s * sy
    a10, a11 = s * fx * sx, c * sy
    u0, v0 = (out_w - 1) / 2.0, (out_h - 1) / 2.0
    return torch.tensor([[a00, a01, cx - a00 * u0 - a01 * v0],
                         [a10, a11, cy - a10 * u0 - a11 * v0]], dtype=torch.float32)


def warp(img, A, out_w, out_h, pad_mode="zeros"):
    """img: [C,H,W] or [B,C,H,W] float; A: [2,3] or [B,2,3]. Bilinear sample at A @ (u,v,1)."""
    single = img.dim() == 3
    if single:
        img, A = img[None], A[None]
    B, C, H, W = img.shape
    dev = img.device
    v, u = torch.meshgrid(torch.arange(out_h, device=dev, dtype=torch.float32),
                          torch.arange(out_w, device=dev, dtype=torch.float32), indexing="ij")
    uv1 = torch.stack([u, v, torch.ones_like(u)], -1).view(1, -1, 3)             # [1,N,3]
    xy = uv1 @ A.to(dev).transpose(1, 2)                                            # [B,N,2]
    gx = xy[..., 0] / max(W - 1, 1) * 2 - 1
    gy = xy[..., 1] / max(H - 1, 1) * 2 - 1
    grid = torch.stack([gx, gy], -1).view(B, out_h, out_w, 2)
    out = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode=pad_mode, align_corners=True)
    return out[0] if single else out


def invert(A):
    """Inverse of a 2x3 affine (input -> output)."""
    M = torch.cat([A, torch.tensor([[0.0, 0.0, 1.0]], dtype=A.dtype)], 0)
    return torch.linalg.inv(M)[:2]


def apply(A, pts):
    """pts [...,2] -> A @ [x,y,1]."""
    return pts @ A[:, :2].T + A[:, 2]


def mask_outside_box(img, box, f=1.0):
    """Zero every source pixel outside the box scaled by f around its center ([C,H,W] float). f<=0: no-op."""
    if f <= 0:
        return img
    _, H, W = img.shape
    x, y, w, h = [float(v) for v in box[:4]]
    cx, cy = x + 0.5 * w, y + 0.5 * h
    x0, x1 = max(0, math.floor(cx - 0.5 * w * f)), min(W, math.ceil(cx + 0.5 * w * f))
    y0, y1 = max(0, math.floor(cy - 0.5 * h * f)), min(H, math.ceil(cy + 0.5 * h * f))
    out = torch.zeros_like(img); out[:, y0:y1, x0:x1] = img[:, y0:y1, x0:x1]
    return out
