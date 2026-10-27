"""Official ViTPose / mmpose 0.24 top-down test pipeline (use_udp=True, post_process='default', flip test,
shift_heatmap=False), ported to torch + numpy for the submission container (no cv2, no mmpose).
Mirrors: TopDownCocoDataset bbox cleaning + _xywh2cs (test mode), TopDownAffine(use_udp=True),
TopDown.forward_test flip averaging, keypoints_from_heatmaps(use_udp=True) = _get_max_preds + post_dark_udp
+ transform_preds(use_udp=True), and the dataset's rescoring (mean of joint scores > vis_thr) x box score."""
import numpy as np
import torch
from t3.geometry import crop_affine, warp

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# flip pairs of configs/_base_/datasets/upar.py (COCO pairs minus the eyes, which are head_bottom/head_top here; ears kept)
FLIP_INDEX = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def clean_box(box, img_w, img_h):
    """TopDownCocoDataset._load_coco_keypoint_annotation_kernel: x2 = min(W-1, x1 + max(0, w-1)). None if degenerate."""
    x, y, w, h = [float(v) for v in box[:4]]
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(img_w - 1.0, x1 + max(0.0, w - 1.0)), min(img_h - 1.0, y1 + max(0.0, h - 1.0))
    if not (x2 > x1 and y2 > y1):
        return None
    return x1, y1, x2 - x1, y2 - y1


def center_scale(box, in_w, in_h, padding):
    """_xywh2cs (test mode): center = xy + wh/2, aspect-fixed size, x padding. Returns center and window size in px (= scale*200)."""
    x, y, w, h = box
    cx, cy = x + w * 0.5, y + h * 0.5
    ar = in_w / in_h
    if w > ar * h:
        h = w / ar
    elif w < ar * h:
        w = h * ar
    return cx, cy, w * padding, h * padding


def _cv2_gauss_kernel(ks):
    """cv2.getGaussianKernel(ks, 0): sigma = 0.3*((ks-1)*0.5 - 1) + 0.8 (ks > 7)."""
    sigma = 0.3 * ((ks - 1) * 0.5 - 1) + 0.8
    x = np.arange(ks, dtype=np.float64) - (ks - 1) / 2
    k = np.exp(-x * x / (2 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def _blur(hm, ks):
    """cv2.GaussianBlur(hm, (ks, ks), 0) per map, BORDER_REFLECT_101 (= numpy 'reflect'). hm [N,K,H,W] float32."""
    k = _cv2_gauss_kernel(ks); r = ks // 2
    p = np.pad(hm, ((0, 0), (0, 0), (0, 0), (r, r)), mode="reflect")
    out = sum(k[i] * p[..., i:i + hm.shape[3]] for i in range(ks))
    p = np.pad(out, ((0, 0), (0, 0), (r, r), (0, 0)), mode="reflect")
    return sum(k[i] * p[..., i:i + hm.shape[2], :] for i in range(ks)).astype(np.float32)


def get_max_preds(hm):
    N, K, _, W = hm.shape
    flat = hm.reshape(N, K, -1)
    idx = np.argmax(flat, 2).reshape(N, K, 1); maxvals = np.amax(flat, 2).reshape(N, K, 1)
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W
    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def post_dark_udp(coords, hm, kernel=11):
    """mmpose.core.evaluation.top_down_eval.post_dark_udp (line-for-line; cv2 blur replaced by _blur)."""
    B, K, H, W = hm.shape
    N = coords.shape[0]
    hm = _blur(hm, kernel)
    np.clip(hm, 0.001, 50, hm)
    np.log(hm, hm)
    pad = np.pad(hm, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge").flatten()
    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = pad[index]; ix1 = pad[index + 1]; iy1 = pad[index + W + 2]
    ix1y1 = pad[index + W + 3]; ix1_y1_ = pad[index - W - 3]
    ix1_ = pad[index - 1]; iy1_ = pad[index - 2 - W]
    dx = 0.5 * (ix1 - ix1_); dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1).reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_; dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords = coords - np.einsum("ijmn,ijnk->ijmk", hessian, derivative).reshape(N, K, 2)
    return coords


def predict_batch_udp(model, images, boxes, cfg, device):
    """images: list of uint8 [3,H,W] tensors (one per box); boxes: [x,y,w,h]. Returns kps [B,17,3] (image px),
    scores [B] (mmpose rescoring, box score 1), maxvals [B,17]. Boxes that mmpose would drop return None in `bad`."""
    in_w, in_h, pad = cfg["in_w"], cfg["in_h"], cfg["padding"]
    crops, cs = [], []
    for img, box in zip(images, boxes):
        cb = clean_box(box, img.shape[2], img.shape[1]) or (float(box[0]), float(box[1]), max(float(box[2]), 1.0), max(float(box[3]), 1.0))
        cx, cy, sw, sh = center_scale(cb, in_w, in_h, pad)
        crops.append(warp(img.to(device).float(), crop_affine(cx, cy, sw, sh, in_w, in_h), in_w, in_h, "zeros"))
        cs.append((cx, cy, sw, sh))
    x = (torch.stack(crops) / 255.0 - MEAN.to(device)) / STD.to(device)
    hm = model(x).float()
    if cfg.get("flip", True):
        hm = 0.5 * (hm + model(x.flip(-1)).float().flip(-1)[:, FLIP_INDEX])
    hm = hm.cpu().numpy().astype(np.float32)
    N, K, Hh, Wh = hm.shape
    preds, maxvals = get_max_preds(hm)
    preds = post_dark_udp(preds, hm.copy(), kernel=cfg.get("udp_kernel", 11))
    cs = np.asarray(cs, np.float64)
    out = np.empty((N, K, 3), np.float32)
    out[..., 0] = preds[..., 0] * cs[:, None, 2] / (Wh - 1.0) + cs[:, None, 0] - cs[:, None, 2] * 0.5
    out[..., 1] = preds[..., 1] * cs[:, None, 3] / (Hh - 1.0) + cs[:, None, 1] - cs[:, None, 3] * 0.5
    out[..., 2] = 1.0
    mv = maxvals[..., 0]; thr = cfg.get("vis_thr", 0.2)
    valid = mv > thr
    score = np.where(valid.sum(1) > 0, (mv * valid).sum(1) / np.maximum(valid.sum(1), 1), 0.0)
    return torch.from_numpy(out), torch.from_numpy(score.astype(np.float32)), torch.from_numpy(mv)
