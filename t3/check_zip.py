"""Run a built zip's run.predict_image on local validation records and compare AP with training-time eval.
python -m t3.check_zip --zip submissions/x.zip --split full --n 2000
--split full = 5% holdout of all domains (Market split by identity), lodo:<domain> = one whole domain."""
import argparse, importlib.util, os, shutil, sys, tempfile, time, zipfile, json
from t3.data import load_all, split
from t3.evaluate import full_report

ap = argparse.ArgumentParser(); ap.add_argument("--zip", required=True); ap.add_argument("--split", default="full"); ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()
work = tempfile.mkdtemp(); zipfile.ZipFile(a.zip).extractall(work)
spec = importlib.util.spec_from_file_location("run_sub", os.path.join(work, "run.py")); run = importlib.util.module_from_spec(spec)
_, va = split(load_all(), a.split, seed=0)
if a.n: va = va[:a.n]
spec.loader.exec_module(run); run.load_model()
t0 = time.time(); preds = {}
for r in va:
    o = run.predict_image(dict(image_id=r["uid"], file_name=r["file_name"], image_path=r["path"], width=r["width"], height=r["height"], boxes=[r["bbox"]], box_ids=[0]))
    preds[r["uid"]] = (o["keypoints"][0], o["scores"][0])
dt = time.time() - t0
rep = full_report(va, preds)
print(f"{len(va)} images, {dt/len(va)*1000:.1f} ms/img")
print(json.dumps({k: {m: round(x, 4) for m, x in v.items()} for k, v in rep.items() if k != "joint_oks"}))
shutil.rmtree(work)
