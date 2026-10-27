"""Wrap an official ViTPose (mmpose) checkpoint as a t3 ckpt for t3.package (decode=udp, padding 1.25 as in the mmpose test pipeline).
python -m t3.mm2ckpt <mmpose epoch_N.pth> <hf_dir from t3.convert_orig> <out ckpt_last.pt> <mmpose config path> [box padding, default 1.25]"""
import hashlib, os, sys, torch
from t3.model import PoseNet

src, hf, out, mmcfg = sys.argv[1:5]; padding = float(sys.argv[5]) if len(sys.argv) > 5 else 1.25
m = PoseNet(hf, in_hw=(256, 192))
sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
cfg = dict(model=hf, in_w=192, in_h=256, padding=padding, pad_mode="zeros", split="all", decode="udp",
           mmpose_ckpt=os.path.basename(src), mmpose_ckpt_sha256=sha, mmpose_config=mmcfg)   # file name + hash only, no local paths
torch.save(dict(cfg=cfg, model=m.state_dict()), out)
print("saved", out, "sha", sha[:16])
