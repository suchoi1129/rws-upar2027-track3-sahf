"""Local COCO-style keypoint evaluation (OKS AP, maxDets=20) mirroring the challenge description."""
import contextlib, io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from t3.common import SIGMAS, EVAL_SLOTS, NAMES


def build_gt(recs):
    images, anns = [], []
    for r in recs:
        images.append(dict(id=r["uid"], width=r["width"], height=r["height"], file_name=r["file_name"]))
        k = r["kps"].copy()
        anns.append(dict(id=r["uid"] + 1, image_id=r["uid"], category_id=1, bbox=r["bbox"], area=r["area"], iscrowd=0,
                         keypoints=k.reshape(-1).tolist(), num_keypoints=int((k[:, 2] > 0).sum())))
    gt = COCO()
    gt.dataset = dict(images=images, annotations=anns, categories=[dict(id=1, name="person")])
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def coco_eval(recs, preds):
    """preds: {uid: (kps[17,3], score)}. Returns dict AP, AP50, AP75, APm, APl."""
    gt = build_gt(recs)
    dts = [dict(image_id=u, category_id=1, keypoints=np.asarray(v[0], float).reshape(-1).tolist(), score=float(v[1]))
           for u, v in preds.items()]
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dts)
        E = COCOeval(gt, dt, "keypoints")
        E.params.kpt_oks_sigmas = SIGMAS
        E.params.maxDets = [20]
        E.evaluate(); E.accumulate(); E.summarize()
    s = E.stats
    return dict(AP=s[0], AP50=s[1], AP75=s[2], APm=s[3], APl=s[4])


def per_joint_oks(recs, preds):
    """Mean per-keypoint OKS term exp(-d^2/(2 s^2 k^2)) over labeled joints, per slot."""
    acc = {j: [] for j in EVAL_SLOTS}
    for r in recs:
        k = np.asarray(preds[r["uid"]][0]); g = r["kps"]; s2 = max(r["area"], 1.0)
        for j in EVAL_SLOTS:
            if g[j, 2] > 0:
                d2 = ((k[j, :2] - g[j, :2]) ** 2).sum()
                acc[j].append(np.exp(-d2 / (2 * s2 * (2 * SIGMAS[j]) ** 2)))
    return {NAMES[j]: float(np.mean(v)) if v else float("nan") for j, v in acc.items()}


def full_report(recs, preds):
    out = {"all": coco_eval(recs, preds)}
    for d in sorted({r["domain"] for r in recs}):
        sub = [r for r in recs if r["domain"] == d]
        out[d] = coco_eval(sub, {r["uid"]: preds[r["uid"]] for r in sub})
    out["joint_oks"] = per_joint_oks(recs, preds)
    return out
