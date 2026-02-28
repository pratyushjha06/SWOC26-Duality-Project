# Results — Semantic Segmentation Experiments (UNet Pipeline)

## Full Outputs

All qualitative results, model checkpoints, and visualizations:

 Google Drive:  
https://drive.google.com/drive/folders/1M8hoq3mhfo9STr3yV1Z9hDuRl_nQ2nyS

---

## Baseline

| Exp ID  | Model               | Augmentation | Train Loss | Val Loss | mIoU   | Dice   | Pixel Acc | Inference |
| ------- | ------------------- | ------------ | ---------- | -------- | ------ | ------ | --------- | --------- |
| Exp_000 | DINOv2-Small + Head | None         | 1.0085     | 1.1079   | 0.2601 | 0.3951 | 0.6845    | 12.78 ms  |

---

## Model-Focused Experiments

| Exp ID  | Model              | Train Loss | Val Loss   | mIoU   | Dice   | Pixel Acc  | Inference |
| ------- | ------------------ | ---------- | ---------- | ------ | ------ | ---------- | --------- |
| Exp_001 | UNet               | **0.4035** | **0.3939** | 0.4366 | 0.5779 | **0.8611** | 1.14 ms   |
| Exp_002 | DeepLabV3-ResNet50 | 0.6435     | 0.6593     | 0.4016 | 0.5576 | 0.7510     | 1.92 ms   |
| Exp_003 | SegNet             | 0.9397     | 0.8980     | 0.2866 | 0.3937 | 0.6897     | 1.02 ms   |

 UNet selected as core architecture (best accuracy + efficiency)

---

## Data-Focused Experiments

| Exp ID  | Augmentation Strategy                  | Val Loss   | mIoU       | Dice       | Pixel Acc  | Inference |
| ------- | -------------------------------------- | ---------- | ---------- | ---------- | ---------- | --------- |
| Exp_004 | Flips + Rotation + Scaling             | 0.6802     | 0.2756     | 0.4477     | 0.7630     | 2.12 ms   |
| Exp_005 | Brightness + Contrast                  | **0.4125** | **0.4348** | **0.5692** | **0.8583** | 3.08 ms   |
| Exp_006 | Flip + Resized Crop                    | 0.6802     | 0.2756     | 0.4477     | 0.7630     | 2.12 ms   |
| Exp_007 | Geometric + Photometric + Noise + Blur | 0.6018     | 0.3081     | 0.4497     | 0.7909     | 2.20 ms   |

 Best augmentation: Brightness + Contrast

---

## Training-Focused Experiments

### Learning Rate Study

| LR   | Val Loss   | mIoU       |
| ---- | ---------- | ---------- |
| 1e-4 | **0.3576** | **0.4746** |
| 3e-4 | 0.4037     | 0.4200     |
| 5e-5 | 0.4315     | 0.4298     |

 Optimal LR: 1e-4

---

### Optimizer Comparison

| Optimizer | Val Loss   | mIoU       |
| --------- | ---------- | ---------- |
| Adam      | **0.4205** | **0.4226** |
| AdamW     | 0.4198     | 0.4213     |
| SGD       | 0.4968     | 0.3852     |

 Adam chosen

---

### Loss Function Study

| Loss Function      | Val Loss   | mIoU   | Dice       | Pixel Acc |
| ------------------ | ---------- | ------ | ---------- | --------- |
| Cross-Entropy      | 0.3435     | 0.5096 | 0.6731     | 0.8764    |
| CE + Dice          | 0.8676     | 0.5318 | 0.6872     | 0.8744    |
| Focal Loss         | **0.0392** | 0.5265 | **0.6926** | 0.8751    |
| Weighted CE + Dice | 0.9179     | 0.4951 | 0.6448     | 0.8407    |

---

### Batch Size Study

| Batch Size   | Val Loss | mIoU       | Pixel Acc  |
| ------------ | -------- | ---------- | ---------- |
| 2            | 0.8513   | 0.5565     | 0.8768     |
| 4 (baseline) | 0.8335   | **0.5611** | **0.8777** |
| 8            | 0.8415   | 0.5552     | 0.8764     |

 Batch Size 4 retained

---

## Final Configuration

**Architecture:** UNet  
**Augmentation:** Brightness + Contrast  
**Optimizer:** Adam  
**Learning Rate:** 1e-4  
**Batch Size:** 4  
**Loss:** Cross-Entropy  

---

## Key Observations

- UNet outperformed heavier architectures while being faster
- Lighting augmentation produced the largest performance gain
- Excessive augmentation degraded segmentation quality
- Training hyperparameters significantly influenced performance
- Model achieves high accuracy with real‑time inference capability
