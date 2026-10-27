"""Build a submission zip from a t3 checkpoint (t3.mm2ckpt): TorchScript export + bundled t3 inference modules.
python -m t3.package --ckpt work/t3_ckpt.pt --out submissions/upar_vitpose_h_pad110_degaug_sahf_9view.zip --half 1 --shifts '8,0;-8,0;0,8;0,-8;6,6;6,-6;-6,6;-6,-6'"""
import argparse, json, os, shutil, tempfile, zipfile
import numpy as np, torch
from t3.model import PoseNet, Ensemble

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = ["__init__.py", "common.py", "geometry.py", "decode.py", "infer.py", "udp.py", "udp_shift.py"]


PRIOR_FILE = os.path.join(ROOT, "t3", "prior_pose.json")


def prior_pose():
    """Box-normalized mean pose of the training annotations (fallback for degenerate boxes). Read from t3/prior_pose.json,
    so packaging needs no data; computed from the training data (and written there) only if that file is missing."""
    if os.path.isfile(PRIOR_FILE):
        return json.load(open(PRIOR_FILE))
    from t3.data import load_all
    recs = load_all(); acc = np.zeros((17, 2)); cnt = np.zeros(17)
    for r in recs:
        x, y, w, h = r["bbox"]; k = r["kps"]; m = k[:, 2] > 0
        acc[m] += np.c_[(k[m, 0] - x) / w, (k[m, 1] - y) / h]; cnt[m] += 1
    pr = acc / np.maximum(cnt, 1)[:, None]
    pr[3] = [0.55, pr[0][1]]; pr[4] = [0.45, pr[0][1]]          # ears unlabeled: place beside the nose
    pr = pr.tolist(); json.dump(pr, open(PRIOR_FILE, "w"))
    return pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--half", type=int, default=1); ap.add_argument("--flip", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prune_moe", type=int, default=1, help="ViTPose+ : keep only the expert selected by dataset_index (smaller zip, same output)")
    ap.add_argument("--weights", default="", help="comma-separated heatmap weights for the ensemble members (default equal)")
    ap.add_argument("--decode", default="", choices=["", "dark", "udp"], help="udp: official ViTPose/mmpose test pipeline (t3/udp.py); default: the ckpt cfg value, else dark")
    ap.add_argument("--tta", default="", help="comma-separated paddings for multi-padding TTA, e.g. 1.0,1.1,1.2")
    ap.add_argument("--shift_px", type=float, default=0, help="udp only: 5-view SAHF with axial shifts of +-d network-input px (t3/udp_shift.py)")
    ap.add_argument("--shifts", default="", help="udp only: explicit shift list 'dx,dy;dx,dy;...' in network-input px (overrides --shift_px), e.g. 9-view")
    a = ap.parse_args()
    models, cfgs = [], []
    for path in a.ckpt.split(","):                       # comma-separated -> heatmap-averaging ensemble
        ck = torch.load(path, map_location="cpu", weights_only=False); cc = ck["cfg"]
        mm = PoseNet(cc["model"], in_hw=(cc["in_h"], cc["in_w"])); mm.load_state_dict(ck["model"]); models.append(mm.eval()); cfgs.append(cc)
        if a.prune_moe and mm.moe:
            idx = int(mm.dataset_index.item())
            for mod in mm.modules():
                if hasattr(mod, "experts") and hasattr(mod, "num_experts"):
                    mod.experts = torch.nn.ModuleList([mod.experts[idx]]); mod.num_experts = 1
            mm.dataset_index.zero_()
    c = cfgs[0]
    assert all((x["in_h"], x["in_w"], x["padding"], x["pad_mode"], x.get("decode", "dark")) == (c["in_h"], c["in_w"], c["padding"], c["pad_mode"], c.get("decode", "dark")) for x in cfgs), "ensemble members must share crop and decode settings"
    m = models[0] if len(models) == 1 else Ensemble(models, [float(v) for v in a.weights.split(",")] if a.weights else None).eval()
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(f"{tmp}/assets"); os.makedirs(f"{tmp}/t3")
        x = torch.randn(4, 3, c["in_h"], c["in_w"])
        with torch.no_grad():
            ref = m(x)
            if a.half:
                # fp16 weights (half the zip size); traced on CUDA, not frozen so .float() still works for CPU fallback
                mh = m.cuda().half()
                ts = torch.jit.trace(mh, x[:1].cuda().half(), strict=False)
                ts.save(f"{tmp}/assets/model.torchscript")
                re = torch.jit.load(f"{tmp}/assets/model.torchscript", map_location="cuda")
                diff = (re(x.cuda().half()).float().cpu() - ref).abs().max().item(); tol = 5e-3
            else:
                ts = torch.jit.trace(m, x[:1], strict=False)
                ts.save(f"{tmp}/assets/model.torchscript")
                diff = (torch.jit.load(f"{tmp}/assets/model.torchscript")(x) - ref).abs().max().item(); tol = 1e-4
        print(f"torchscript max|diff| ({'fp16' if a.half else 'fp32'}) = {diff:.2e}")
        assert diff < tol, "TorchScript output differs from eager model"
        cfg = dict(in_w=c["in_w"], in_h=c["in_h"], padding=c["padding"], pad_mode=c["pad_mode"], flip=bool(a.flip),
                   dark_ks=11, dark_sigma=2.0, decode=a.decode or c.get("decode", "dark"), half=bool(a.half), batch=a.batch, prior_pose=prior_pose(),
                   tta_pads=[float(v) for v in a.tta.split(",")] if a.tta else None,
                   shifts=([[float(v.split(",")[0]), float(v.split(",")[1]), "px"] for v in a.shifts.split(";")] if a.shifts else
                           [[a.shift_px, 0, "px"], [-a.shift_px, 0, "px"], [0, a.shift_px, "px"], [0, -a.shift_px, "px"]] if a.shift_px else None),
                   ens_weights=a.weights or None, source_ckpt=[os.path.relpath(p, ROOT) for p in a.ckpt.split(",")], source_model=[x["model"] for x in cfgs], train_cfg=cfgs)
        json.dump(cfg, open(f"{tmp}/assets/config.json", "w"), indent=1)
        shutil.copy(f"{ROOT}/t3/submission_template/run.py", f"{tmp}/run.py")
        shutil.copy(f"{ROOT}/t3/submission_template/metadata.yaml", f"{tmp}/metadata.yaml")   # = starter kit example metadata.yaml
        for f in BUNDLE:
            shutil.copy(f"{ROOT}/t3/{f}", f"{tmp}/t3/{f}")
        out = os.path.join(ROOT, a.out); os.makedirs(os.path.dirname(out), exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for d, _, fs in os.walk(tmp):
                for f in fs:
                    p = os.path.join(d, f); z.write(p, os.path.relpath(p, tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    assert "run.py" in names and "metadata.yaml" in names
    print("wrote", out, f"{os.path.getsize(out)/1e6:.0f} MB", sorted(names))


if __name__ == "__main__":
    main()
