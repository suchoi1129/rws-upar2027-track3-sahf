# Shift-Aligned Heatmap Fusion for UPAR 2027 Track 3 (2D Human Pose Estimation)

Code, configurations and instructions to reproduce our entry to Track 3 of the UPAR Challenge at the RWS workshop, WACV 2027.

The system is ViTPose-H (classic decoder, UDP) fine-tuned on the UPAR training data with two changes to the training
pipeline, and a training-free inference method:

* **Crop preprocessing for tight pedestrian crops.** Box padding 1.25 → 1.10 and the official one-off train-time center
  perturbation disabled (`bbox_padding`, `fixed_center_jitter` in the dataset config).
* **Degradation augmentation** (training only, on the full image before cropping): low resolution (p 0.25, ×0.35–0.9),
  Gaussian blur (p 0.15, kernel 3 or 5), JPEG (p 0.3, quality 20–75), Gaussian noise (p 0.1, σ ∈ [2, 8] on 0–255).
* **Shift-Aligned Heatmap Fusion (SAHF)** at test time: the same model is applied to the center crop and to eight crops
  shifted by about half a ViT patch (±8 px axial, (±6, ±6) px diagonal, in 256×192 input pixels), each with flip testing;
  the heatmaps are translated back to the grid of the center view by the known offsets, averaged, and decoded once with UDP.

## Results (hidden test set, Codabench)

| Configuration | AP | AP50 | AP75 | AP_M | AP_L |
|---|---|---|---|---|---|
| ViTPose-H, official recipe (padding 1.25) | 0.9032 | 0.9704 | 0.9386 | 0.8755 | 0.6935 |
| + padding 1.10 | 0.9040 | 0.9706 | 0.9368 | 0.8770 | 0.6948 |
| + degradation augmentation | 0.9053 | 0.9733 | 0.9398 | 0.8774 | 0.6962 |
| + SAHF, 5 views | 0.9080 | 0.9735 | 0.9400 | 0.8807 | 0.6974 |
| **+ SAHF, 9 views (submitted)** | **0.9082** | **0.9736** | **0.9401** | **0.8810** | **0.6975** |

## Repository layout

```
.
├── data/                          # from the starter kit (step 2), not in git
│   ├── annotations/
│   │   ├── specialization/        # train_{market,pa100k,peta}.json
│   │   └── lodo/                  # written by scripts/make_splits.py
│   ├── market/
│   ├── pa100k/
│   └── peta/
├── weights/                       # vitpose-h.pth, upar_vitpose_h_pad110_degaug.pth (step 3), not in git
├── work/, submissions/            # training runs and built zips, not in git
├── vitpose/                       # official ViTPose (@ c050ed2) with our changes
│   └── configs/upar/              # final/, full_data/, validation/
├── t3/                            # conversion, packaging and the submission's inference code
├── scripts/                       # make_splits.py, train.sh, build_submission.sh
├── docker/                        # Dockerfile.train, Dockerfile.infer
├── logs/                          # training log of the submitted model
└── requirements/                  # train_freeze.txt, package.txt
```

Our changes to the official ViTPose code are three files (`mmpose/datasets/pipelines/upar_degrade.py`,
`mmpose/datasets/pipelines/__init__.py` and the top-down dataset base class
`mmpose/datasets/datasets/base/kpt_2d_sview_rgb_img_top_down_dataset.py`) and the new configs (`configs/upar/`, and `configs/_base_/datasets/upar.py` for the keypoint definition of the
challenge); without
the new config options (`bbox_padding`, `fixed_center_jitter`) the code behaves as the official release. SAHF is
implemented in `t3/udp_shift.py`.

## 1. Setup

### 1.1 Requirements

Check these three points before starting; everything else (Python, PyTorch, the CUDA runtimes, mmcv) is inside the
Docker images, so no CUDA toolkit has to be installed on the host.

1. **NVIDIA driver with CUDA 12.1 or newer** (Linux driver 530 or newer): `nvidia-smi` must show "CUDA Version: 12.1"
   or higher. This covers both images: the inference image uses the CUDA 12.1 runtime and the training image the CUDA
   11.1 runtime, each bundled with its PyTorch.
2. **GPUs visible in Docker** (NVIDIA Container Toolkit): `docker run --rm --gpus all ubuntu:20.04 nvidia-smi` must list
   the GPUs.
3. **GPUs for the step you run.**
   * Inference (building and running the submission): one GPU.
   * Training: we trained on 8 × A100-SXM4-80GB with 64 images per GPU (batch 512), using about 53 GB of memory per GPU.
     The training environment (PyTorch 1.9.0, CUDA 11.1) runs on GPUs up to the Ampere generation (e.g. V100, A100); it
     is not built for newer generations such as H100.

**Choose the GPUs for your machine.** The commands below use `--gpus all` for training and `--gpus '"device=0"'` for
inference; change the device list (or `CUDA_VISIBLE_DEVICES` without Docker) and the number of training GPUs
(`GPUS=<n>` in `scripts/train.sh`, default 8) as needed. With fewer GPUs the total batch size changes, so results can
differ from ours.

### 1.2 Docker

The code is copied into both images; data, weights and outputs are mounted from this repository.

#### 1.2.1 Training image (`docker/Dockerfile.train`)

The ViTPose environment of `requirements/train_freeze.txt` (Python 3.8, PyTorch 1.9.0 + CUDA 11.1, mmcv-full 1.3.9),
installed with uv on Ubuntu 20.04. Used for training and validation (step 4).

```bash
docker build -f docker/Dockerfile.train -t upar-sahf-train .
```

#### 1.2.2 Inference image (`docker/Dockerfile.infer`)

The challenge container `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` (Python 3.11, PyTorch 2.4.1 + CUDA 12.1) plus
`requirements/package.txt`. Used for downloads (steps 2 and 3), conversion, packaging and inference (step 5).

```bash
docker build -f docker/Dockerfile.infer -t upar-sahf-infer .
```

#### 1.2.3 Common run options (folders and user)

Required: every later `docker run` uses these options. They connect the folders of this repository to the
container (`$M`) and run the container as your user (`$U`).

```bash
mkdir -p data weights work submissions
R=/workspace/upar2027-track3-sahf
U="--user $(id -u):$(id -g) -e HOME=/tmp -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro"   # outputs owned by you
M="-v $PWD/data:$R/data -v $PWD/weights:$R/weights -v $PWD/work:$R/work -v $PWD/submissions:$R/submissions"
```

All later commands use `$R`, `$U` and `$M` from this block; run it from the repository root, again in every new shell.
`$U` makes the outputs belong to your user; without it they belong to root. Keep the `/etc/passwd` mount with `--user`:
mmcv looks up the user name and fails for an unknown user id.

### 1.3 Without Docker (uv)

Both environments are plain virtual environments made with [uv](https://docs.astral.sh/uv/).

#### 1.3.1 Training environment (`.venv-train`)

Same installation as `docker/Dockerfile.train`; `requirements/train_freeze.txt` is the exact package list we trained with.

```bash
IDX="--extra-index-url https://download.pytorch.org/whl/cu111 --index-strategy unsafe-best-match"
uv venv -p 3.8 .venv-train
uv pip install -p .venv-train/bin/python torch==1.9.0+cu111 torchvision==0.10.0+cu111 $IDX
uv pip install -p .venv-train/bin/python pip "setuptools<70" wheel numpy==1.23.5         # build tools for chumpy / xtcocotools
uv pip install -p .venv-train/bin/python --no-build-isolation -r requirements/train_freeze.txt $IDX   # includes `-e ./vitpose`
```

#### 1.3.2 Inference environment (`.venv-t3`)

The packages of `docker/Dockerfile.infer`, with PyTorch 2.4.1 + CUDA 12.1 installed from `requirements/package.txt`
instead of the base image.

```bash
uv venv -p 3.11 .venv-t3
uv pip install -p .venv-t3/bin/python -r requirements/package.txt
```

#### 1.3.3 Running the commands of this README

Activate the matching environment (`.venv-train` for `upar-sahf-train`, `.venv-t3` for `upar-sahf-infer`) and run the
part after the image name from the repository root.

## 2. Data

Download the training data with `download_datasets.py` of the official starter kit into `data/`, then build the
training files:

```bash
git clone https://github.com/MickaelCormier/UPAR-Challenge-2027-Track3.git ../upar-starter-kit
docker run --rm $U $M -v $(realpath ../upar-starter-kit):/starter:ro upar-sahf-infer \
    python /starter/download_datasets.py --data-dir data          # -> data/{market,pa100k,peta}/, data/annotations/specialization/
docker run --rm $U $M upar-sahf-infer python scripts/make_splits.py   # -> data/annotations/lodo/*.json
```

Both scripts run in the inference image (or `.venv-t3`), which contains gdown, numpy and tqdm; the system Python of recent
Linux distributions refuses `pip install`. The tools in `t3/` read `data/`; set `UPAR_DATA=<dir>` to use another location.

`make_splits.py` writes `all_train.json` (all three domains, used for the submitted model), the leave-one-domain-out
(LODO) files `{market,pa100k,peta}_{train,val}.json`, and the 80/20 split `split82_{train,val}.json` (seed 0; Market-1501
split by person identity). Instance counts:

| Source | Instances | LODO train | 8:2 train | 8:2 val |
|---|---|---|---|---|
| Market-1501 | 12,923 | 29,813 | 10,565 | 2,358 |
| PA-100K | 19,499 | 23,237 | 15,623 | 3,876 |
| PETA | 10,314 | 32,422 | 8,257 | 2,057 |
| UPAR (total) | 42,736 | – | 34,445 | 8,291 |

## 3. Weights

| File | Use | Place it at | sha256 | Download |
|---|---|---|---|---|
| `vitpose-h.pth` | initialization for training (official COCO-trained ViTPose-H, classic decoder) | `weights/vitpose-h.pth` | `092b137f18fb92ec0c569dd9d003468e4a31fbd9248fe4523cd9f10a15ad17a6` | [ViTPose release (OneDrive)](https://1drv.ms/u/s!AimBgYV7JjTlgShLMI-kkmvNfF_h?e=dEhGHe) |
| `upar_vitpose_h_pad110_degaug.pth` | our trained model (padding 1.10, degradation augmentation; mmpose checkpoint, weights only, = `epoch_36.pth` of step 4) | `weights/upar_vitpose_h_pad110_degaug.pth` | `82a30d85ef04ee76c9d35a3377333d8cf4165d699e462d7c319e49c771b85e7b` | [Google Drive](https://drive.google.com/drive/folders/18VlqQVr0upUlgo5WwzN-Qxy6GgAAFINP) |
| `upar_vitpose_h_pad110_degaug_sahf_9view.zip` | the submitted Codabench package (9-view SAHF), built from the file above with step 5 | `submissions/upar_vitpose_h_pad110_degaug_sahf_9view.zip` | `646e53c8a375631eb1e00caf703a307456078300093ef6a2451d96535ea8dbb2` | [Google Drive](https://drive.google.com/drive/folders/18VlqQVr0upUlgo5WwzN-Qxy6GgAAFINP) |

Our two files are in one Google Drive folder (https://drive.google.com/drive/folders/18VlqQVr0upUlgo5WwzN-Qxy6GgAAFINP):

```bash
docker run --rm $U -v $PWD:/w -w /w upar-sahf-infer gdown --folder https://drive.google.com/drive/folders/18VlqQVr0upUlgo5WwzN-Qxy6GgAAFINP   # -> UPAR_TRACK3_SAHF/
sha256sum UPAR_TRACK3_SAHF/*                                                # compare with the table above
mv UPAR_TRACK3_SAHF/*.pth weights/ && mv UPAR_TRACK3_SAHF/*.zip submissions/
```

`vitpose-h.pth` (needed for training only) is downloaded from the ViTPose release in a browser. No other pretrained weights
or external data are used.

## 4. Training

### 4.1 Train the submitted model

Needs the data (step 2) and `weights/vitpose-h.pth` (step 3).

```bash
docker run --rm --gpus all --ipc=host $U $M upar-sahf-train \
    bash scripts/train.sh final/upar_vitpose_h_p110_deg.py work/final --no-validate
```

8 GPUs × 64 images (batch 512), 36 epochs, AdamW (lr 5e-4, weight decay 0.1, layer-wise decay 0.85, gradient clipping
1.0), 500-iteration linear warm-up, ×0.1 learning-rate steps at epochs 29 and 34, drop path 0.55, seed 0. Every reported
number uses the final checkpoint `work/final/epoch_36.pth`; there is no checkpoint selection; build the submission from it as in 5.1. The training log of the submitted model
(environment, configuration, and per-epoch learning rate, loss, accuracy, memory and time) is
`logs/20261018_040631.log`, unchanged; its paths and run name are those of our training machine. For another number of GPUs
set `-e GPUS=<n>` (and the matching `--gpus` device list); see 1.1.

### 4.2 Validation protocols

Each `configs/upar/validation/*.py` trains one model on a LODO fold or on the 8:2 split and evaluates it with mmpose at
epoch 36 (single view, flip test), e.g.

```bash
docker run --rm --gpus all --ipc=host $U $M upar-sahf-train bash scripts/train.sh validation/lodo_peta_p110_deg.py work/lodo_peta
```

To evaluate SAHF on a held-out domain, build a zip from the fold model as in 5.1 and run
`python -m t3.check_zip --zip <zip> --split lodo:peta` in the inference image (likewise `lodo:market`, `lodo:pa100k`).

## 5. Inference

### 5.1 Build the submission zip

From our released weights (step 3) or from your own training run (`work/final/epoch_36.pth`, step 4):

```bash
docker run --rm --gpus '"device=0"' $U $M upar-sahf-infer \
    bash scripts/build_submission.sh weights/upar_vitpose_h_pad110_degaug.pth submissions/rebuilt.zip
# or: ... bash scripts/build_submission.sh work/final/epoch_36.pth submissions/my_sahf_9view.zip
```

This converts the mmpose checkpoint to the Hugging Face ViTPose implementation, exports an FP16 TorchScript model and
writes a zip with `run.py` at its root (the challenge API: `load_model()`, `predict_image(sample)`); inside the challenge
container the zip needs only torch, numpy and Pillow. The nine views are set by `--shifts` in
`scripts/build_submission.sh`. `t3.convert_orig` reads the model configuration of `usyd-community/vitpose-base` from the
Hugging Face Hub once (internet access needed for this step only; all weights come from the checkpoint). If `data/` is
present, the zip is then run on 2,000 images of the 5% holdout of `t3.data.split`; these images are part of the training
data of the submitted model, so this is a functional check, not a validation score.

### 5.2 Run the submission locally (as the challenge server does)

Codabench runs `run.py` of the zip on the hidden test set. `t3.predict` makes the same calls locally on any COCO-style
annotation file, e.g. the hidden test set or the UPAR training data:

```bash
docker run --rm --gpus '"device=0"' $U $M upar-sahf-infer python -m t3.predict \
    --zip submissions/upar_vitpose_h_pad110_degaug_sahf_9view.zip --ann <annotations.json> --img-root <image dir> --out submissions/preds.json
```

`t3.predict` calls `run.py` the way the challenge ingestion does (`load_model()`, then `predict_image()` once per image
with all its boxes) and writes COCO keypoint results (`image_id`, `category_id`, `keypoints`, `score`, `bbox`). It reads
only `images[]` and the boxes of `annotations[]`, so it also runs on files without keypoint labels. Paths must be inside
the mounted folders (e.g. `data/annotations/specialization/train_peta.json` with `--img-root data`).

### 5.3 Verify against the submitted package

The zip rebuilt in 5.1 gives the same keypoints and scores as the submitted package from Google Drive (step 3). This
check needs no UPAR data; it uses the four COCO test images shipped with ViTPose:

```bash
for z in rebuilt upar_vitpose_h_pad110_degaug_sahf_9view; do
  docker run --rm --gpus '"device=0"' $U $M upar-sahf-infer python -m t3.predict --zip submissions/$z.zip \
      --ann vitpose/tests/data/coco/test_coco.json --img-root vitpose/tests/data/coco --out submissions/preds_$z.json
done
python3 -c "import json; a, b = (json.load(open(f'submissions/preds_{z}.json')) for z in ('rebuilt', 'upar_vitpose_h_pad110_degaug_sahf_9view')); \
print(len(a), 'boxes, max keypoint difference:', max(abs(x - y) for p, q in zip(a, b) for x, y in zip(p['keypoints'], q['keypoints']))) "
```

The expected output is `14 boxes, max keypoint difference: 0.0`.

## Acknowledgements and licenses

Our code is released under the Apache-2.0 license (`LICENSE`). `vitpose/` is ViTPose (Apache-2.0, see `vitpose/LICENSE`). The UPAR data is described in Specker et al., WACV 2023, and
Cormier et al., WACVW 2023; the source datasets (Market-1501, PA-100K, PETA) keep their own licenses.
