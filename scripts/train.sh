#!/bin/bash
# Train ViTPose-H with one of the UPAR configs, 64 images per GPU, seed 0.
# usage: [GPUS=<n>] bash scripts/train.sh <config under vitpose/configs/upar/> <work dir> [extra tools/train.py args]
#   e.g. bash scripts/train.sh final/upar_vitpose_h_p110_deg.py work/final --no-validate
# GPUs: set GPUS to the number of GPUs of your machine (default 8, as in our runs: 8 x 64 = batch 512) and choose them with
# CUDA_VISIBLE_DEVICES or docker --gpus. The batch per GPU stays 64, so with another GPUS the total batch changes.
set -e
CFG=$1; WORK=$(realpath -m "$2"); shift 2
cd "$(dirname "$0")/../vitpose"
PYTHONPATH=$PWD:$PYTHONPATH bash tools/dist_train.sh configs/upar/$CFG ${GPUS:-8} --work-dir "$WORK" --seed 0 "$@"
