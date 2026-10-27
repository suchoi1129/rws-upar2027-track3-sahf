"""Run a submission zip on any COCO-style annotation file (images + person boxes) and write COCO keypoint results.
The zip's run.py is called exactly as the challenge ingestion does: load_model(), then predict_image() once per image
with all boxes of that image. Ground-truth keypoints, if present, are not read.
python -m t3.predict --zip submissions/upar_vitpose_h_pad110_degaug_sahf_9view.zip --ann data/annotations/specialization/train_peta.json \
                     --img-root data --out preds_peta.json
Output: [{"image_id", "category_id": 1, "keypoints": [x, y, v] * 17, "score", "bbox"}] in the order of the annotations."""
import argparse, importlib.util, json, os, shutil, tempfile, time, zipfile
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--zip", required=True, help="submission zip (run.py at its root)")
ap.add_argument("--ann", required=True, help="COCO-style json with images[] and annotations[] (bbox = [x, y, w, h])")
ap.add_argument("--img-root", required=True, help="directory that images[].file_name is relative to")
ap.add_argument("--out", required=True, help="output json (COCO keypoint results)")
a = ap.parse_args()

d = json.load(open(a.ann))
boxes = defaultdict(list)
for ann in d["annotations"]:
    boxes[ann["image_id"]].append(ann)
work = tempfile.mkdtemp(); zipfile.ZipFile(a.zip).extractall(work)
spec = importlib.util.spec_from_file_location("run_sub", os.path.join(work, "run.py")); run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run); run.load_model()
t0 = time.time(); res = []
for im in d["images"]:
    anns = boxes.get(im["id"], [])
    if not anns:
        continue
    o = run.predict_image(dict(image_id=im["id"], file_name=im["file_name"], image_path=os.path.join(a.img_root, im["file_name"]),
                               width=im["width"], height=im["height"], boxes=[x["bbox"] for x in anns], box_ids=[x["id"] for x in anns]))
    for x, k, s in zip(anns, o["keypoints"], o["scores"]):
        res.append(dict(image_id=im["id"], category_id=1, keypoints=[float(v) for p in k for v in p], score=float(s), bbox=x["bbox"]))
dt = time.time() - t0
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(res, open(a.out, "w"))
print(f"{len(res)} boxes in {len(boxes)} images, {dt / max(len(boxes), 1) * 1000:.1f} ms/img -> {a.out}")
shutil.rmtree(work)
