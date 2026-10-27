#!/bin/bash
# mmpose checkpoint -> submission zip (9-view SAHF), then a quick local check of the zip.
# usage: bash scripts/build_submission.sh <epoch_36.pth> <out.zip> [config, default final/upar_vitpose_h_p110_deg.py]
# GPU: uses the first visible GPU (one is enough); choose it with CUDA_VISIBLE_DEVICES or docker --gpus '"device=N"'.
set -e
CKPT=$(realpath "$1"); OUT=$(realpath -m "$2"); CFG=${3:-final/upar_vitpose_h_p110_deg.py}
cd "$(dirname "$0")/.."
W=work/package; mkdir -p $W
python -m t3.convert_orig "$CKPT" $W/hf_model                                           # mmpose weights -> HF transformers ViTPose
python -m t3.mm2ckpt "$CKPT" $W/hf_model $W/t3_ckpt.pt vitpose/configs/upar/$CFG 1.1   # + crop / decode settings (padding 1.10, UDP)
python -m t3.package --ckpt $W/t3_ckpt.pt --out "$OUT" --half 1 \
    --shifts "8,0;-8,0;0,8;0,-8;6,6;6,-6;-6,6;-6,-6"                                    # center view + 8 shifted views
if [ -d data/annotations/specialization ]; then
    python -m t3.check_zip --zip "$OUT" --split full --n 2000                           # runs run.py from the zip on local images
else
    echo "data/annotations/specialization not found: skipping the local check of $OUT"
fi
