"""Annotation loading. Uses the 3 per-domain json files (train_upar_all.json has broken PETA paths)."""
import json, os
import numpy as np
from t3.common import DATA, DOMAINS


def load_domain(dom):
    a = json.load(open(f"{DATA}/annotations/specialization/train_{dom}.json"))
    imgs = {i["id"]: i for i in a["images"]}
    out = []
    for ann in a["annotations"]:
        im = imgs[ann["image_id"]]
        k = np.asarray(ann["keypoints"], np.float32).reshape(17, 3)
        fn = im["file_name"]
        pid = os.path.basename(fn).split("_")[0] if dom == "market" else None
        out.append(dict(domain=dom, path=f"{DATA}/{fn}", file_name=fn, image_id=ann["image_id"], ann_id=ann["id"],
                        width=im["width"], height=im["height"], bbox=[float(v) for v in ann["bbox"]],
                        area=float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])), kps=k,
                        num_kp=int(ann.get("num_keypoints", (k[:, 2] > 0).sum())),
                        attrs=np.asarray(json.loads(ann["attrs"]) if isinstance(ann.get("attrs"), str) else ann.get("attrs", []), np.int8)))
    return out


def load_all(domains=DOMAINS):
    recs = []
    for d in domains:
        r = load_domain(d)
        missing = [x["path"] for x in r if not os.path.isfile(x["path"])]
        assert not missing, f"{d}: {len(missing)} missing image files, e.g. {missing[:3]}"
        recs += r
    # globally unique ids (image ids may collide across domains)
    for i, r in enumerate(recs):
        r["uid"] = i
    return recs


def split(recs, mode, seed=0, val_frac=0.05):
    """mode: 'lodo:<domain>' -> (train = other domains, val = held-out domain)
             'full'          -> 95/5 split over all domains (Market split by person id).
             'all'           -> train on everything; val = the 'full' holdout (monitoring only, seen in training)."""
    if mode == "all":
        tr, va = split(recs, "full", seed, val_frac)
        return tr + va, va
    if mode.startswith("lodo:"):
        held = mode.split(":")[1]
        return [r for r in recs if r["domain"] != held], [r for r in recs if r["domain"] == held]
    rng = np.random.default_rng(seed)
    pids = sorted({r["path"].split("/")[-1].split("_")[0] for r in recs if r["domain"] == "market"})
    val_pid = set(rng.choice(pids, int(len(pids) * val_frac), replace=False))
    tr, va = [], []
    for r in recs:
        if r["domain"] == "market":
            (va if r["path"].split("/")[-1].split("_")[0] in val_pid else tr).append(r)
        else:
            (va if rng.random() < val_frac else tr).append(r)
    return tr, va
