"""Annotation files for training and validation, built from the three per-domain json files of the starter kit
(data/annotations/specialization/train_{market,pa100k,peta}.json). Writes data/annotations/lodo/:
  all_train.json                 all three domains merged (training of the submitted model)
  {d}_train.json / {d}_val.json  leave-one-domain-out (LODO): train = the other two domains, val = domain d unchanged
  split82_train.json / _val.json 80/20 split of every domain (seed 0); Market-1501 is split by person identity
Image and annotation ids are renumbered when domains are merged (they collide across domains); nothing else is changed.
usage: python scripts/make_splits.py [--data data]"""
import argparse, collections, json, os, shutil
import numpy as np

D = ["market", "pa100k", "peta"]


def merge(jsons):
    out = dict(images=[], annotations=[], categories=jsons[0]["categories"]); iid = aid = 0
    for j in jsons:
        m = {}
        for im in j["images"]:
            iid += 1; m[im["id"]] = iid; out["images"].append(dict(im, id=iid))
        for a in j["annotations"]:
            aid += 1; out["annotations"].append(dict(a, id=aid, image_id=m[a["image_id"]]))
    return out


def split82(d):
    """80/20 split of the merged json (seed 0): Market-1501 by person id, PA-100K / PETA per instance."""
    im = {i["id"]: i for i in d["images"]}; rng = np.random.default_rng(0)
    dom = lambda f: f.split("/")[0]
    mk = sorted({im[a["image_id"]]["file_name"].split("/")[-1].split("_")[0] for a in d["annotations"] if dom(im[a["image_id"]]["file_name"]) == "market"})
    val_pid = set(rng.choice(mk, int(round(len(mk) * 0.2)), replace=False))
    tr, va = [], []
    for a in d["annotations"]:
        f = im[a["image_id"]]["file_name"]
        isval = f.split("/")[-1].split("_")[0] in val_pid if dom(f) == "market" else rng.random() < 0.2
        (va if isval else tr).append(a)
    def sub(anns):
        ids = {a["image_id"] for a in anns}
        return dict(d, images=[i for i in d["images"] if i["id"] in ids], annotations=anns)
    cnt = lambda anns: dict(collections.Counter(dom(im[a["image_id"]]["file_name"]) for a in anns))
    print("split82 train", len(tr), cnt(tr), "| val", len(va), cnt(va), "| market ids", len(mk), "val ids", len(val_pid))
    return sub(tr), sub(va)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--out", default="", help="output folder (default <data>/annotations/lodo)"); a = ap.parse_args()
    S = f"{a.data}/annotations/specialization"; O = a.out or f"{a.data}/annotations/lodo"; os.makedirs(O, exist_ok=True)
    J = {d: json.load(open(f"{S}/train_{d}.json")) for d in D}
    for d in D:
        shutil.copy(f"{S}/train_{d}.json", f"{O}/{d}_val.json")
        out = merge([J[o] for o in D if o != d]); json.dump(out, open(f"{O}/{d}_train.json", "w"))
        print(d, "train imgs", len(out["images"]), "anns", len(out["annotations"]), "| val imgs", len(J[d]["images"]), "anns", len(J[d]["annotations"]))
    allj = merge([J[o] for o in D]); json.dump(allj, open(f"{O}/all_train.json", "w"))
    print("all train imgs", len(allj["images"]), "anns", len(allj["annotations"]))
    tr, va = split82(allj); json.dump(tr, open(f"{O}/split82_train.json", "w")); json.dump(va, open(f"{O}/split82_val.json", "w"))


if __name__ == "__main__":
    main()
