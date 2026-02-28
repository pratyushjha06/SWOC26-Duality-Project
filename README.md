# Off‑Road Semantic Segmentation (SWOC26 – Duality AI)

Pixel‑wise segmentation of synthetic off‑road desert scenes into 10 terrain / object classes using a UNet‑based pipeline and a bunch of controlled experiments (models, data, and training tricks).

> Repo: [GitHub - pratyushjha06/SWOC26-Duality-Project](https://github.com/pratyushjha06/SWOC26-Duality-Project)  
> Full experiments / notebooks / heavy artifacts (Drive): https://drive.google.com/drive/folders/1am0BU9onk4TCyApVagrpZL9uXfL8TTYb

---

## 1. Problem

Given an RGB image from a synthetic desert environment, predict a semantic label for **every pixel** from the following 10 classes:

> Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Flowers, Logs, Rocks, Landscape, Sky

The dataset is highly imbalanced (Landscape / Sky dominate; Logs / Flowers / Rocks are rare), so a lot of the work is about **making the model not ignore the rare stuff**.

If you want the full story (motivation, dataset, challenges, final numbers), read:

- `Report.md`

---

## 2. Repository Layout

High‑level structure of this repo:

```text
SWOC26-Duality-Project/
├── data/                  # (not tracked) Train / Val / Test
├── scripts/
│   ├── dataset.py         # MaskDataset + transforms + ID→class mapping
│   ├── train_segmentation.py
│   ├── train_final.py     # final UNet training entrypoint
│   ├── test_segmentation.py
│   └── visualize.py       # quick visualizations
├── models/
│   ├── models.md          # ALL model/data/training experiments (UNet, DeepLab, etc.)
│   └── .gitkeep
├── results/
│   ├── baseline/          # baseline masks, visualizations, per-class metrics
│   ├── final_model/       # best_cases, failures, metrics.json, class_iou.json
│   └── .gitkeep
|   └── RESULT.md
├── Report.md              # technical report (problem → final model)
└── README.md              # you are here
```

Detailed descriptions of **every experiment** (all models, data aug, loss functions, LRs, batch sizes) live in:

- `models/models.md`

I’m not repeating that content here on purpose.

---

## 3. Setup

### 3.1 Environment

```bash
git clone https://github.com/pratyushjha06/SWOC26-Duality-Project.git
cd SWOC26-Duality-Project

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Core deps: `torch`, `torchvision`, `albumentations`, `numpy`, `tqdm`, `matplotlib`.

### 3.2 Data Placement

Expected layout:

```text
data/
  Train/
    Color_Images/
    Segmentation/
  Val/
    Color_Images/
    Segmentation/
  Test/
    Color_Images/
```

Update paths in the scripts if your structure is slightly different (check `scripts/dataset.py`).

---

## 4. How to Run

### 4.1 Train a Baseline UNet

This uses the “baseline” config (simpler training, no fancy loss) that matches your early experiments.

```bash
python scripts/train_segmentation.py \
  --train-dir data/Train \
  --val-dir data/Val \
  --save-dir results/baseline
```

Outputs (inside `results/baseline/`):

- `masks/` – predicted masks
- `masks_color/` – colorized predictions
- `visualizations/` – side‑by‑side input / GT / prediction
- `evaluation_metrics` (json/txt)
- `per_class_metrics` (plots)

### 4.2 Train the Final UNet

This uses the best configuration found through all the T1/T2/T3/T5 experiments (LR, optimizer, loss, batch size, aug). The exact settings are documented in `models/models.md` and `Report.md`, but you don’t need to read them to run this.

```bash
python scripts/train_final.py \
  --train-dir data/Train \
  --val-dir data/Val \
  --save-dir results/final_model
```

Checkpoints are saved into `results/final_model` (or `models/final_unet` if you point `--save-dir` there).

### 4.3 Evaluate a Trained Model

```bash
python scripts/test_segmentation.py \
  --checkpoint path/to/best_model.pth \
  --val-dir data/Val \
  --save-dir results/final_model
```

You’ll get:

- `metrics.json` – overall mIoU, Dice, pixel accuracy, timing
- `class_iou.json` – per‑class IoU
- `best_cases/` – examples where the model crushed it
- `failures/` – examples where it messed up

`visualize.py` is for quick qualitative checks during development.

---

## 5. What’s Inside the Experiments (Pointer Only)

All the heavy experimentation is already written in **one place**:

- `models/models.md`

That file contains:

- Model baselines: DINOv2 + ConvNeXt head, UNet, DeepLabV3, SegNet
- Data‑focused experiments:
  - Albumentations geometric aug
  - Brightness/contrast ColorJitter
  - Strong appearance aug
  - Geometric‑only aug
- Training‑focused experiments:
  - Learning‑rate sweep (T1)
  - Optimizer comparison (Adam, SGD, AdamW) (T2)
  - Loss functions (CE, CE+Dice, Focal, Weighted CE+Dice) (T3, T3_B)
  - Batch size sweep (2, 4, 8) (T5)
- Final model configuration + metrics

If someone wants **numbers, plots, or ablation details**, they go there instead of scrolling through a monster README.

For a narrative / report style view (for evaluation or hackathon submission), use:

- `Report.md`

---

## 6. Final Model (TL;DR)

What we actually ended up using:

- **Architecture:** UNet (3‑channel input, 10‑class output)
- **Augmentation:** brightness/contrast jitter + light geometric transforms
- **Optimizer:** Adam, LR 1e‑4
- **Loss:** Weighted Cross‑Entropy + Dice
- **Batch size:** 4
- **Selection:** best validation mean IoU

Validation set (317 images):

- Mean IoU ≈ **0.61**
- Dice Score ≈ **0.72**
- Pixel Accuracy ≈ **0.89**
- Inference ≈ **2.2 ms / image** on a T4‑class GPU

Exact numbers + per‑class breakdown are in `results/final_model/metrics.json` and `class_iou.json`, as well as `Report.md`.

---

## 7. Extra Stuff

- All raw experiment notebooks + every single result (per‑experiment folders, visualizations, checkpoints) live in the Drive: `https://drive.google.com/drive/folders/1am0BU9onk4TCyApVagrpZL9uXfL8TTYb`
- `Experiment_Tracker.xlsx` has a spreadsheet‑style log of runs and metrics.

---

## 8. Acknowledgements

- Duality AI & SWOC26 organizers for the dataset and challenge.
- PyTorch, Albumentations, and the wider OSS community.

If you want to plug in a new model or loss, open a PR or fork this and go wild.