# Off‑Road Semantic Segmentation – Documentation Index

Central index for all documentation and experiments in this repository.

---

## 1. High‑Level README

For a quick overview of the project, goals, repo structure, and how to run the final UNet:

- `README.md`

This is the main entry point for anyone landing on the repo.

---

## 2. Model Experiments

Detailed descriptions, configs, and metrics for all **model‑focused, data‑focused, and training‑focused** experiments:

- `models/models.md`

This file covers:

- DINOv2 + ConvNeXt baseline  
- UNet, DeepLabV3, SegNet  
- Data‑augmentation experiments (Exp_001–Exp_004)  
- Training ablations (learning rate, optimizer, loss, batch size, class weights)

---

## 3. Technical Report

For a narrative, paper‑style explanation of the final approach:

- `Report.md`

Contents:

- Problem statement and dataset description  
- Preprocessing and augmentation pipeline  
- Final UNet architecture and training strategy  
- Quantitative results and qualitative discussion  
- Challenges, solutions, and future work

---

## 4. Code & Scripts

Implementation entry points:

- `scripts/dataset.py` – dataset + transforms  
- `scripts/train_final_unet.py` – train final model  
- `scripts/eval_final_unet.py` – evaluate + save metrics/visualizations  
- Additional helper scripts live in the same folder.

---

## 5. Experiments & Results

Local result folders (lightweight artifacts):

- `results/baseline/` – early baselines, masks, per‑class metrics, visualizations  
- `results/final_model/` – best_cases, failures, `metrics.json`, `class_iou.json`

Full notebooks, checkpoints, histories, and visualizations are stored in the shared Drive:

- **Main Drive folder:**  
  `https://drive.google.com/drive/folders/1am0BU9onk4TCyApVagrpZL9uXfL8TTYb`

---

## 6. Notebooks

Exploratory and experiment notebooks (for reproducibility):

- `notebooks/ModelExp1_UNet.ipynb`  
- `notebooks/Exp_002_Brightness_Contrast.ipynb`  
- `notebooks/T1_-Learning-Rate.ipynb`  
- `notebooks/T2_-Optimization.ipynb`  
- `notebooks/T3_-Loss-functions.ipynb`  
- `notebooks/T3_B_WeightedCE_Dice.ipynb`  
- `notebooks/T4_-Batch-size.ipynb`  
- `notebooks/Final_UNet_Training.ipynb`

Use these to re‑run or extend individual experiments.

---

## 7. Quick Pointers

- Want **just the final model**? See `models/final_unet/` and `results/final_model/`.  
- Want **all experiment details**? Read `models/models.md`.  
- Want a **short narrative for reports/hackathons**? Use `Report.md`.  
- Want to **run code**? Start from `README.md` and the scripts in `scripts/`.