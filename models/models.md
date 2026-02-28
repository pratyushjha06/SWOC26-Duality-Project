# Off‑Road Semantic Segmentation Experiments

This document summarizes a series of **model‑focused**, **data‑focused**, and **training‑focused** experiments conducted on an off‑road semantic segmentation dataset (synthetic desert scenes, 10 classes).

- Model‑focused: Baselines and architecture comparisons.
- Data‑focused: Augmentation and dataset‑centric improvements.
- Training‑focused: Optimization, loss functions, and batch size.
- Final model: Consolidated UNet configuration and evaluation.

Drive Link: https://drive.google.com/drive/folders/12ZGMumDkAF3UxvzZ4jnw8tFKKtnU2-TK

---

## 1. Baseline Model and Dataset

### 1.1 Baseline: DINOv2 + ConvNeXt‑Style Head

**Phase:** Baseline Training (Feature Extractor + Lightweight Head)  
**Backbone:** DINOv2 ViT‑S/14 (frozen)  
**Head:** ConvNeXt‑style convolutional segmentation head  
**Task:** Multi‑class semantic segmentation (10 classes)  
**Dataset:** Synthetic off‑road desert scenes (RGB + masks)

#### 1.1.1 Model Overview

The model combines a frozen DINOv2 Vision Transformer backbone (ViT‑S/14) with a lightweight convolutional segmentation head. The backbone extracts patch‑level semantic features, which the head decodes into pixel‑wise class predictions.

#### 1.1.2 Backbone: DINOv2 (ViT‑S/14)

- Self‑supervised Vision Transformer pretrained on large‑scale data.  
- Variant: ViT‑S/14, output grid: 34 × 19 = 646 tokens, embedding dim: 384.  
- Used in evaluation mode (weights frozen) as a generic feature extractor.

#### 1.1.3 Segmentation Head

- ConvNeXt‑style convolutional head.  
- Reshapes tokens into a spatial feature map.  
- Applies depthwise convolution blocks.  
- Outputs multi‑class logits for 10 semantic classes.  
- Total trainable parameters in the head: ~2.43M.

#### 1.1.4 Training Configuration

- Device: NVIDIA Tesla T4 GPU  
- Batch size: 8  
- Learning rate: 1e‑4  
- Optimizer: SGD with momentum 0.9  
- Loss: Cross‑Entropy Loss  
- Epochs: 10  
- Input resolution: 266 × 476

#### 1.1.5 Dataset

- Training samples: 2,857  
- Validation samples: 317  
- Images resized and normalized, masks converted to class indices (0–9).

#### 1.1.6 Training Performance (Best Epoch)

Best validation metrics at epoch 10:

- Mean IoU: 0.2674  
- Dice score: 0.4014  
- Pixel accuracy: 0.6845  
- Validation loss: 0.8854

Metrics improved steadily across epochs, indicating stable convergence.

#### 1.1.7 Final Validation & Inference

Final evaluation on the validation set:

- Mean IoU: 0.2601  
- Dice score: 0.3951  
- Pixel accuracy: 0.6845  
- Average inference time: 12.78 ms per image

Predictions and visualizations were saved for qualitative analysis.

#### 1.1.8 Strengths

- Strong pretrained feature extractor.  
- Fast inference relative to heavy backbones.  
- Stable training.  
- Low parameter count in the segmentation head.

#### 1.1.9 Limitations

- Backbone not fine‑tuned for domain‑specific features.  
- Moderate IoU compared to stronger segmentation architectures.  
- Struggles with small or rare classes.  
- Patch‑based representation can reduce spatial precision.

---

### 1.2 Dataset Class Distribution (Train & Val)

This section summarizes the class frequency distribution of the training and validation sets after mapping raw IDs to 10 semantic classes.

#### 1.2.1 Class Mapping

Raw mask IDs are mapped to class indices as follows:

| Raw ID | Class Index | Class Name     |
| ------ | ----------- | -------------- |
| 100    | 0           | Trees          |
| 200    | 1           | Lush Bushes    |
| 300    | 2           | Dry Grass      |
| 500    | 3           | Dry Bushes     |
| 550    | 4           | Ground Clutter |
| 600    | 5           | Flowers        |
| 700    | 6           | Logs           |
| 800    | 7           | Rocks          |
| 7100   | 8           | Landscape      |
| 10000  | 9           | Sky            |

#### 1.2.2 Pixel‑Level Class Frequencies

**Train set class counts:**

`[ 52,331,525  87,892,776 279,430,843 16,268,713 65,082,995 41,585,811  1,153,995 17,743,187 362,120,221 557,458,734 ]`

**Val set class counts:**

`[ 6,685,175  9,887,005 31,778,064 1,806,117 6,966,231 4,001,940 109,285 1,995,285 39,059,907 62,043,791 ]`

**Train class frequencies (% of pixels):**

- 0 – Trees: 3.53%  
- 1 – Lush Bushes: 5.93%  
- 2 – Dry Grass: 18.87%  
- 3 – Dry Bushes: 1.10%  
- 4 – Ground Clutter: 4.39%  
- 5 – Flowers: 2.81%  
- 6 – Logs: 0.08%  
- 7 – Rocks: 1.20%  
- 8 – Landscape: 24.45%  
- 9 – Sky: 37.64%

**Val class frequencies (% of pixels):**

- 0 – Trees: 4.07%  
- 1 – Lush Bushes: 6.02%  
- 2 – Dry Grass: 19.34%  
- 3 – Dry Bushes: 1.10%  
- 4 – Ground Clutter: 4.24%  
- 5 – Flowers: 2.44%  
- 6 – Logs: 0.07%  
- 7 – Rocks: 1.21%  
- 8 – Landscape: 23.77%  
- 9 – Sky: 37.75%

These distributions highlight heavy dominance of **Sky** and **Landscape**, and strong rarity of **Logs**, **Flowers**, and **Rocks**, motivating class‑balanced losses and careful evaluation of minority‑class performance.

---

## 2. Model‑Focused Experiments

### 2.1 Model Experiment 1: UNet Baseline

**Experiment ID:** ModelExp1_UNet  
**Architecture:** UNet (encoder–decoder CNN)  
**Task:** Multi‑class semantic segmentation (10 classes)

#### 2.1.1 Model Overview

A classical UNet is evaluated for segmentation of synthetic off‑road scenes. UNet is a fully convolutional encoder–decoder with skip connections, designed to preserve spatial detail for pixel‑level predictions.

#### 2.1.2 Dataset and Preprocessing

- RGB images + masks with 10 classes.  
- Image size: 266 × 476, normalized with ImageNet stats.  
- Masks resized and converted to integer class labels.  
- Training samples: 2,857  
- Validation samples: 317

#### 2.1.3 Architecture Details

- Contracting path: Double Conv blocks (Conv → BatchNorm → ReLU) + MaxPool.  
- Expanding path: Transposed Conv for upsampling + skip connections from encoder.  
- Final layer: 1×1 conv producing logits for 10 classes.

#### 2.1.4 Training Configuration

- Device: CUDA‑enabled GPU  
- Optimizer: Adam  
- Learning rate: 1e‑4  
- Loss: Cross‑Entropy Loss  
- Batch size: 4  
- Epochs: 5  
- Best model selected by minimum validation loss.

#### 2.1.5 Training Performance

- Training loss decreased steadily.  
- Validation performance improved consistently without severe overfitting.  
- Final validation loss: 0.6009  
- Validation pixel accuracy: 0.7629

#### 2.1.6 Validation Metrics

- Mean IoU: 0.4070  
- Mean Dice score: 0.5346  
- Pixel accuracy: 0.7633

#### 2.1.7 Inference Performance

- Average inference time: ~2.28 ms per image.

#### 2.1.8 Qualitative Analysis

The model captures large regions (sky, terrain) well, with remaining errors on small objects and visually similar classes.

#### 2.1.9 Strengths

- Higher segmentation accuracy than baseline DINOv2 head.  
- Preserves spatial details via skip connections.  
- Fast inference.  
- Stable training dynamics.

#### 2.1.10 Limitations

- Higher parameter count than lightweight heads.  
- Struggles with very small or rare classes.  
- Performance sensitive to dataset size and diversity.

---

### 2.2 Model Experiment 2: DeepLabV3 (ResNet‑50)

**Experiment ID:** ModelExp2_DeepLabV3  
**Architecture:** DeepLabV3 with ResNet‑50 backbone  
**Task:** Multi‑class semantic segmentation (10 classes)

#### 2.2.1 Model Overview

DeepLabV3 uses atrous convolutions and an Atrous Spatial Pyramid Pooling (ASPP) module to capture multi‑scale context. A ResNet‑50 backbone pretrained on COCO is used as the feature encoder.

#### 2.2.2 Dataset and Preprocessing

- RGB images with pixel‑wise labels for 10 classes.  
- Image size: 266 × 476, ImageNet normalization.  
- Masks resized and converted to integer class IDs.  
- Training samples: 2,857  
- Validation samples: 317

#### 2.2.3 Architecture Details

- Encoder: ResNet‑50.  
- ASPP module for multi‑scale receptive fields.  
- Final classifier: 1×1 conv head for 10 classes.

#### 2.2.4 Training Configuration

- Device: CUDA‑enabled GPU  
- Optimizer: Adam  
- Learning rate: 1e‑4  
- Loss: Cross‑Entropy Loss  
- Batch size: 4  
- Epochs: 5  
- Best model selected by validation loss.

#### 2.2.5 Training Performance

- Training loss decreased steadily.  
- Validation performance improved until epoch 4, then stabilized.  
- Best validation loss: 0.6399 (epoch 4)  
- Validation pixel accuracy: 0.7506

#### 2.2.6 Validation Metrics

- Mean IoU: 0.4016  
- Mean Dice score: 0.5576  
- Pixel accuracy: 0.7510

#### 2.2.7 Inference Performance

- Average inference time: ~1.92 ms per image.

#### 2.2.8 Qualitative Analysis

DeepLabV3 captures boundaries and large structures effectively, with ASPP helping across multiple scales. Small objects and rare classes remain challenging.

#### 2.2.9 Strengths

- Strong multi‑scale feature extraction.  
- Competitive segmentation accuracy.  
- Efficient inference.  
- Benefits from a pretrained ResNet‑50 backbone.

#### 2.2.10 Limitations

- Higher training compute cost than simpler models.  
- May require more epochs for optimal performance.  
- Still challenged by extremely small or infrequent classes.

---

### 2.3 Model Experiment 3: SegNet

**Experiment ID:** ModelExp3_SegNet  
**Architecture:** SegNet (encoder–decoder with MaxUnpooling)  
**Task:** Multi‑class semantic segmentation (10 classes)

#### 2.3.1 Model Overview

SegNet is a fully convolutional encoder–decoder architecture using pooling indices from the encoder to guide decoder upsampling. This preserves spatial structure without storing full feature maps.

#### 2.3.2 Dataset and Preprocessing

- RGB images with pixel‑level annotations (10 classes).  
- Image size: 266 × 476, ImageNet normalization.  
- Masks resized and converted to integer labels.  
- Training samples: 2,857  
- Validation samples: 317

#### 2.3.3 Architecture Details

- VGG‑style encoder with five convolutional blocks (Conv + BN + ReLU + MaxPool).  
- Decoder mirrors encoder using MaxUnpool2d with stored pooling indices.  
- Final layer: logits for 10 classes.

#### 2.3.4 Training Configuration

- Device: CUDA‑enabled GPU  
- Optimizer: Adam  
- Learning rate: 1e‑4  
- Loss: Cross‑Entropy Loss  
- Batch size: 4  
- Epochs: 5  
- Best model selected by validation loss.

#### 2.3.5 Training Performance

- Training loss decreased, validation improved until epoch 4 then showed mild overfitting.  
- Best validation loss: 0.8326 (epoch 4)  
- Validation pixel accuracy: 0.6902

#### 2.3.6 Validation Metrics

- Mean IoU: 0.2866  
- Mean Dice score: 0.3937  
- Pixel accuracy: 0.6897

#### 2.3.7 Inference Performance

- Average inference time: ~1.02 ms per image (fastest among evaluated models).

#### 2.3.8 Qualitative Analysis

SegNet performs well on large homogeneous regions but struggles on fine details and complex boundaries, sometimes confusing similar‑texture classes.

#### 2.3.9 Strengths

- Very fast inference.  
- Memory‑efficient decoding using pooling indices.  
- Stable training.  
- Suitable for real‑time scenarios.

#### 2.3.10 Limitations

- Lower accuracy than UNet and DeepLabV3.  
- Difficulties with fine‑grained structures.  
- Limited context due to lack of explicit multi‑scale modules.

---

## 3. Data‑Focused Experiments

All experiments in this section use UNet variants while modifying only the data pipeline.

### 3.1 Exp‑001: Albumentations Data Augmentation (UNet)

**Experiment ID:** Exp_001_Data_Aug  
**Architecture:** UNet (same as baseline)  
**Focus:** Data augmentation for better generalization  
**Task:** Multi‑class semantic segmentation (10 classes)

#### 3.1.1 Experiment Overview

This experiment studies advanced data augmentation using Albumentations while keeping the architecture unchanged. The goal is to improve robustness, reduce overfitting, and enhance generalization.

#### 3.1.2 Dataset and Augmentation Strategy

- RGB images with pixel‑wise labels (10 classes).  
- Training augmentations:  
  - Resize to 266 × 476  
  - Random rotation (±15°)  
  - Horizontal flip (p = 0.5)  
  - ImageNet normalization  
- Validation: resize + normalization only.

#### 3.1.3 Model Architecture

Identical UNet encoder–decoder with skip connections as in the baseline. Any performance changes are attributable to augmentation rather than architecture.

#### 3.1.4 Training Configuration

- Device: CUDA‑enabled GPU  
- Optimizer: Adam  
- Learning rate: 1e‑4  
- Loss: Cross‑Entropy Loss  
- Batch size: 8 (increased from baseline)  
- Epochs: 5  
- Best model selected by validation loss.

#### 3.1.5 Training Performance

- Training loss decreased consistently.  
- Validation performance improved steadily, with reduced overfitting vs baseline UNet.  
- Final validation loss: 0.3939  
- Validation pixel accuracy: 0.8608

#### 3.1.6 Validation Metrics

- Mean IoU: 0.4366  
- Mean Dice score: 0.5779  
- Pixel accuracy: 0.8611

#### 3.1.7 Inference Performance

- Average inference time: ~1.14 ms per image.

#### 3.1.8 Qualitative Analysis

Boundary delineation improves, with better robustness to orientation changes and scene layout variations.

#### 3.1.9 Strengths

- Significant accuracy gains without changing architecture.  
- Improved generalization to unseen orientations.  
- Reduced overfitting.  
- Maintains fast inference.

#### 3.1.10 Limitations

- Results still hinge on training data diversity.  
- Augmentation does not fully address rare‑class imbalance.  
- Further tuning may yield additional improvements.

---

### 3.2 Exp‑002: Brightness & Contrast (UNet + ColorJitter)

**Focus:** Photometric augmentation (brightness/contrast)  
**Architecture:** UNet (same as previous)

#### 3.2.1 Objective

Improve robustness to illumination changes by applying ColorJitter‑based brightness and contrast augmentation during training.

#### 3.2.2 Dataset

- Training samples: 2,857  
- Validation samples: 317  
- Classes: 10  
- Input resolution: 266 × 476

#### 3.2.3 Data Preprocessing and Augmentation

- Training augmentations:  
  - Resize to 266 × 476  
  - Random brightness adjustment (±20%)  
  - Random contrast adjustment (±20%)  
  - ImageNet normalization  
- Validation: resize + normalization only.  
- Masks resized with nearest‑neighbor interpolation.

#### 3.2.4 Model Architecture

- UNet encoder–decoder with skip connections.  
- Input channels: 3, output classes: 10.  
- Architecture identical to baseline.

#### 3.2.5 Training Configuration

- Loss: CrossEntropyLoss  
- Optimizer: Adam  
- Learning rate: 1e‑4  
- Batch size: 4  
- Epochs: 5  
- Device: GPU (CUDA)

#### 3.2.6 Training Results

Training converged steadily, with decreasing validation loss and improving pixel accuracy under illumination variability.

#### 3.2.7 Final Evaluation Metrics

- Mean IoU: 0.4348  
- Mean Dice score: 0.5692  
- Pixel accuracy: 0.8583

#### 3.2.8 Inference Performance

- Average inference time: 3.08 ms per image.

#### 3.2.9 Qualitative Results

The model becomes more stable under lighting variations, reducing misclassification in shadowed or low‑contrast regions.

#### 3.2.10 Strengths

- Increased robustness to brightness/contrast changes.  
- Maintains high segmentation accuracy.  
- Simple photometric augmentation proves effective.  
- No architectural modification required.

#### 3.2.11 Limitations

- Slight inference‑time increase vs the original baseline.  
- Does not handle geometric variability.  
- Smaller performance gain than some geometric/combined augmentation setups.

#### 3.2.12 Conclusion

Brightness and contrast augmentation improves robustness to illumination while preserving overall performance, showing the value of data‑centric improvements alongside architecture tweaks.

---

### 3.3 Exp‑003: Geometric Augmentation (UNet + Flips & Crops)

#### 3.3.1 Objective

Enhance generalization by adding geometric variability (flips and resized crops) while keeping the UNet architecture fixed.

#### 3.3.2 Dataset

- Training samples: 2,857  
- Validation samples: 317  
- Classes: 10  
- Input resolution: 266 × 476

#### 3.3.3 Data Augmentation Strategy

- Training augmentations:  
  - Resize to 266 × 476  
  - Random horizontal flip (p = 0.5)  
  - Random resized crop (scale 0.9–1.0)  
  - ImageNet normalization  
- Validation: resize + normalization only.  
- Masks resized via nearest‑neighbor.

#### 3.3.4 Model Architecture

UNet encoder–decoder with skip connections, 3 input channels, 10 output classes, identical to baseline.

#### 3.3.5 Training Configuration

- Loss: CrossEntropyLoss  
- Optimizer: Adam  
- Learning rate: 1e‑4  
- Batch size: 4  
- Epochs: 5  
- Device: GPU (CUDA)

#### 3.3.6 Training Results

Validation performance improved gradually, suggesting reduced overfitting to original spatial configurations.

#### 3.3.7 Final Evaluation Metrics

- Mean IoU: 0.2756  
- Mean Dice score: 0.4477  
- Pixel accuracy: 0.7630

#### 3.3.8 Inference Performance

- Average inference time: 2.12 ms per image.

#### 3.3.9 Qualitative Results

Predictions show improved tolerance to flipped orientations and minor scale changes, though gains are smaller than with photometric or more advanced augmentation strategies.

#### 3.3.10 Strengths

- Improved robustness to viewpoint variations.  
- Helps mitigate spatial overfitting.  
- Simple to implement.  
- No extra inference‑time cost.

#### 3.3.11 Limitations

- Lower accuracy than photometric augmentation experiments.  
- Limited benefit due to narrow crop scale range.  
- Does not address illumination variability.

#### 3.3.12 Conclusion

Geometric augmentation provides moderate spatial robustness but is less effective than other augmentation strategies; combining geometric and photometric methods may yield better overall performance.

---

### 3.4 Exp‑004: Strong Appearance Augmentation (UNet)

#### 3.4.1 Objective

Improve robustness to appearance variations (lighting, color, noise, blur) via aggressive photometric augmentation while keeping validation data clean.

#### 3.4.2 Dataset

- Training samples: 2,857  
- Validation samples: 317  
- Classes: 10  
- Input resolution: 266 × 476  
- Task: multi‑class semantic segmentation

#### 3.4.3 Data Augmentation Strategy

Albumentations is used to apply strong appearance transformations during training:

- Horizontal flipping  
- Random rotation (±20°)  
- Strong ColorJitter (brightness, contrast, saturation, hue)  
- Random gamma correction  
- Gaussian noise  
- Gaussian blur  
- Image normalization

Validation data is only resized and normalized.

#### 3.4.4 Model Architecture

Standard UNet:

- Encoder–decoder structure with skip connections.  
- Double Conv blocks (Conv + BN + ReLU).  
- Transposed convolutions for upsampling.  
- Final 1×1 conv for class prediction.  
- Architecture constant across experiments.

#### 3.4.5 Training Configuration

- Optimizer: Adam  
- Learning rate: 1e‑4  
- Loss: CrossEntropyLoss  
- Batch size: 4  
- Epochs: 5  
- Device: GPU (CUDA)  
- Best model chosen by lowest validation loss.

#### 3.4.6 Quantitative Results

- Mean IoU: 0.3081  
- Mean Dice score: 0.4497  
- Pixel accuracy: 0.7909  
- Average inference time: 2.20 ms per image

#### 3.4.7 Analysis

Appearance augmentation improves robustness relative to purely geometric augmentation (Exp‑003) but underperforms more moderate strategies like Exp‑001 and Exp‑002. Excessively strong perturbations can distort crucial semantic cues (color/texture) needed for class discrimination.

#### 3.4.8 Visual Evaluation

The model captures major scene components but struggles with fine boundaries and small objects under heavy augmentation.

#### 3.4.9 Conclusion

Strong appearance‑based augmentation increases robustness but can degrade segmentation accuracy when overly aggressive. A balanced augmentation strategy performs best.

#### 3.4.10 Key Insight

Augmentation strength must match dataset and task characteristics; moderate photometric changes help, while excessive distortion can remove discriminative features.

---

## 4. Training‑Focused Experiments

These experiments keep the data pipeline (often UNet + ColorJitter) largely fixed and modify training hyperparameters or loss functions.

### 4.1 Exp‑005: Learning Rate Sensitivity (UNet + ColorJitter)

#### 4.1.1 Objective

Study how different learning rates affect convergence and final performance of the UNet with brightness/contrast augmentation (ColorJitter).

#### 4.1.2 Experimental Setup

- Model: UNet (encoder–decoder with skip connections)  
- Loss: CrossEntropyLoss  
- Optimizer: Adam  
- Learning rates:  
  - 1e‑4 (T1A)  
  - 3e‑4 (T1B)  
  - 5e‑4 (T1C)  
- Epochs: 8  
- Batch size: 4  
- Device: GPU (CUDA)  
- Data: same ColorJitter pipeline as Exp‑002 (±20% brightness/contrast).

#### 4.1.3 Quantitative Results (Best per LR)

- **LR = 1e‑4 (T1A):**  
  - Best validation loss: 0.3576  
  - Mean IoU: 0.4746  
  - Pixel accuracy: 0.8723  
  - Inference time: 2.20 ms/image  
- **LR = 3e‑4 (T1B):**  
  - Best validation loss: 0.3993  
  - Mean IoU: 0.4441  
  - Pixel accuracy: 0.8720  
- **LR = 5e‑4 (T1C):**  
  - Best validation loss: 0.4315  
  - Mean IoU: 0.4298  
  - Pixel accuracy: 0.8723

#### 4.1.4 Analysis

A learning rate of 1e‑4 yields the lowest validation loss and highest IoU, with stable training and better qualitative segmentations. Larger LRs converge faster initially but overshoot the optimum, resulting in higher final loss and lower IoU.

#### 4.1.5 Conclusion

For this UNet + ColorJitter configuration, a conservative LR of 1e‑4 is preferred and adopted as default in later experiments.

---

### 4.2 Exp‑006: Optimizer Comparison (Adam vs SGD vs AdamW)

#### 4.2.1 Objective

Compare optimizers in terms of convergence speed, IoU, and pixel accuracy on the off‑road segmentation task.

#### 4.2.2 Experimental Setup

Common settings:

- Model: UNet (baseline architecture)  
- Loss: CrossEntropyLoss  
- Batch size: 4  
- Epochs: 4  
- Device: GPU (CUDA)  
- Data: 2,857 train, 317 validation, resized to 266 × 476, normalized.

Optimizer‑specific:

- Adam: LR = 1e‑4  
- SGD: LR = 0.01, momentum 0.9  
- AdamW: LR = 1e‑4, weight decay 1e‑4  

Best model per optimizer chosen by validation loss.

#### 4.2.3 Results Summary

- **Adam:**  
  - Best validation loss: 0.4205  
  - Mean IoU: 0.4226  
  - Pixel accuracy: 0.8589  
- **SGD:**  
  - Best validation loss: 0.4968  
  - Mean IoU: 0.3852  
  - Pixel accuracy: 0.8304  
- **AdamW:**  
  - Best validation loss: 0.4198  
  - Mean IoU: 0.4213  
  - Pixel accuracy: 0.8538

#### 4.2.4 Analysis

Adam and AdamW show very similar validation loss and IoU, with AdamW giving slightly better regularization over time. SGD converges but underperforms in both IoU and pixel accuracy within the limited 4‑epoch budget.

#### 4.2.5 Conclusion

Adaptive optimizers (Adam, AdamW) are better suited for this setup. Adam is kept as the default for its strong performance and simplicity, with AdamW as an alternative when stronger regularization is desired.

---

### 4.3 Exp‑007: Loss Function Ablation (CE vs CE+Dice vs Focal)

#### 4.3.1 Objective

Assess how different loss functions address class imbalance and boundary quality for the ColorJitter‑augmented UNet.

#### 4.3.2 Experimental Setup

- Model: UNet (same as Exp‑002)  
- Data: ColorJitter pipeline (brightness/contrast ±20%)  
- Optimizer: Adam (LR 1e‑4)  
- Batch size: 4  
- Epochs: 10  
- Loss variants:  
  - `ce`: CrossEntropyLoss  
  - `cedice`: CE + Dice loss  
  - `focal`: Focal loss (α = 0.25, γ = 2.0)

#### 4.3.3 Final Metrics (Best Model / Loss)

- **CE:**  
  - Mean IoU: 0.5096  
  - Mean Dice: 0.6731  
  - Pixel accuracy: 0.8764  
- **CE + Dice:**  
  - Mean IoU: 0.5319  
  - Mean Dice: 0.6872  
  - Pixel accuracy: 0.8744  
- **Focal:**  
  - Mean IoU: 0.5265  
  - Mean Dice: 0.6926  
  - Pixel accuracy: 0.8751

#### 4.3.4 Analysis

Both CE + Dice and Focal loss improve IoU and Dice over plain CE, validating the benefit of overlap‑aware and hard‑example‑focused losses. CE + Dice achieves the best IoU, while Focal yields the highest Dice, indicating sharper boundaries and better handling of minority regions.

#### 4.3.5 Conclusion

For this dataset, CE + Dice offers the best IoU and a strong overall metric balance; it is recommended when computation allows. Focal loss is a good alternative when emphasizing hard and rare examples.

---

### 4.4 Exp‑008: Class‑Balanced Loss (Weighted CE + Dice)

#### 4.4.1 Objective

Mitigate class imbalance by up‑weighting rare classes (e.g., logs, flowers, rocks) via class‑balanced loss, and measure impact on global metrics.

#### 4.4.2 Experimental Setup

- Model: UNet (same as previous)  
- Data: ColorJitter‑augmented training set (brightness/contrast ±20%)  
- Optimizer: Adam (LR 1e‑4)  
- Loss: Weighted CrossEntropy + Dice  
- Class weights: inverse of training frequencies, normalized to mean 1  
- Batch size: 4  
- Epochs: 10

#### 4.4.3 Quantitative Results

- Mean IoU: 0.4951  
- Mean Dice: 0.6448  
- Pixel accuracy: 0.8407

These are slightly lower than the best CE + Dice / Focal configurations on global metrics, but qualitatively minority classes receive better treatment.

#### 4.4.4 Analysis

Class‑balanced loss trades some overall IoU and pixel accuracy for improved rare‑class performance—valuable in safety‑critical or rare‑object contexts. Inverse‑frequency weighting helps but may require tuning to avoid over‑penalizing noisy labels.

#### 4.4.5 Conclusion

Weighted CE + Dice is useful when recall on rare classes matters more than aggregate metrics; moderated weighting or focal‑style reweighting could further refine this trade‑off.

---

### 4.5 Exp‑009: Batch Size vs Performance (UNet + ColorJitter)

#### 4.5.1 Objective

Explore the effect of batch size on optimization dynamics, IoU, Dice, and inference, using a ColorJitter‑augmented UNet and CE+Dice training.

#### 4.5.2 Experimental Setup

- Model: UNet (same as Exp‑007)  
- Data: brightness/contrast augmentation (Exp‑002)  
- Loss: CE + Dice (`cedice`)  
- Optimizer: Adam, LR scaled with batch size:  
  - bs = 2 → LR = 0.7 × 1e‑4  
  - bs = 4 → LR = 1.0 × 1e‑4  
  - bs = 8 → LR = 1.3 × 1e‑4  
- Epochs: 10  
- Device: GPU (CUDA)

#### 4.5.3 Final Metrics (Best per Batch Size)

- **Batch size 2:**  
  - Mean IoU: 0.5565  
  - Mean Dice: 0.7214  
  - Pixel accuracy: 0.8768  
- **Batch size 4:**  
  - Mean IoU: 0.5611  
  - Mean Dice: 0.7211  
  - Pixel accuracy: 0.8777  
- **Batch size 8:**  
  - Mean IoU: 0.5552  
  - Mean Dice: 0.7176  
  - Pixel accuracy: 0.8764

#### 4.5.4 Analysis

All batch sizes perform similarly, with batch size 4 achieving the highest IoU and pixel accuracy by a small margin. Smaller batches add gradient noise and slightly slower throughput, while larger batches need higher LR to converge but do not surpass the medium batch.

#### 4.5.5 Conclusion

A batch size of 4 provides the best balance between stability, performance, and compute efficiency and is used as the default in most experiments.

---

## 5. Final UNet Model

This section describes the final chosen UNet configuration, trained by combining the best findings from data‑, model‑, and training‑focused experiments.

### 5.1 Final Configuration Summary

- **Architecture:** UNet (DoubleConv encoder–decoder with skip connections).  
- **Input / Output:** 3‑channel RGB input, 10‑class logits output.  
- **Input resolution:** 266 × 476.  
- **Loss:** CE + Dice (`ce_dice`) as selected from loss ablation.  
- **Optimizer:** Adam.  
- **Base learning rate:** 1e‑4 (from LR sensitivity study).  
- **Batch size:** 4 (from batch size vs performance study).  
- **Epochs:** 20 (Phase‑5 final training).  
- **Experiment name:** `Final_UNet_Phase5`.  
- **Model checkpoint:** `models/final_unet/best_model_final.pth`.  
- **Results directory:** `results/final_unet/`.

### 5.2 Data Pipeline

**Training transforms (image):**

- Resize to 266 × 476.  
- ColorJitter with brightness=0.2, contrast=0.2 (from Exp‑002).  
- ToTensor.  
- Normalize with ImageNet mean and std.

**Validation transforms (image):**

- Resize to 266 × 476.  
- ToTensor.  
- Normalize with ImageNet mean and std.

**Mask transforms:**

- Resize to 266 × 476 using nearest‑neighbor interpolation.  
- ToTensor, followed by conversion to integer class indices 0–9.

Training and validation sets are identical to previous experiments: 2,857 training images and 317 validation images.

### 5.3 Model Architecture (UNet)

- **Encoder:**  
  
  - Down1: DoubleConv(3 → 64) + MaxPool  
  - Down2: DoubleConv(64 → 128) + MaxPool  
  - Down3: DoubleConv(128 → 256) + MaxPool  
  - Down4: DoubleConv(256 → 512) + MaxPool  

- **Bottleneck:**  
  
  - DoubleConv(512 → 1024)

- **Decoder:**  
  
  - Up4: ConvTranspose2d(1024 → 512) + concat with encoder4 + DoubleConv(1024 → 512)  
  - Up3: ConvTranspose2d(512 → 256) + concat with encoder3 + DoubleConv(512 → 256)  
  - Up2: ConvTranspose2d(256 → 128) + concat with encoder2 + DoubleConv(256 → 128)  
  - Up1: ConvTranspose2d(128 → 64) + concat with encoder1 + DoubleConv(128 → 64)

- **Classifier head:**  
  
  - 1×1 conv: 64 → 10 classes.

Spatial misalignments between upsampled features and skip connections are resolved via bilinear interpolation before concatenation, ensuring matching spatial dimensions.

### 5.4 Training and Evaluation Protocol

- **Training loop:**  
  
  - For each epoch: train on full training set, then evaluate on validation set.  
  - Loss per batch: `CE + DiceLoss` over 10 classes.  
  - IoU computed per‑class and averaged (mean IoU) for validation monitoring.

- **Best‑model criterion:**  
  
  - Track validation mean IoU each epoch.  
  - Save checkpoint `best_model_final.pth` whenever validation IoU improves.

- **Logged history:**  
  
  - `train_loss[epoch]`  
  - `val_loss[epoch]`  
  - `val_iou[epoch]`  
    stored in `results/final_unet/train_val_history_phase5.json`.

### 5.5 Final Model Evaluation

Using the separate evaluation notebook, the final UNet checkpoint `best_model_final.pth` was evaluated on the full validation set (317 images).

**Aggregate metrics:**

- Mean IoU: **0.6082**
- Mean Dice: **0.7185**
- Pixel accuracy: **0.8890**

These results confirm that the final UNet with CE+Dice loss, ColorJitter augmentation, LR = 1e‑4, and batch size 4 significantly outperforms the earlier baselines and intermediate experiments, both in overlap‑based metrics (IoU/Dice) and overall pixel accuracy.

---
