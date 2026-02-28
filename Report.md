# Off‑Road Semantic Scene Segmentation Using UNet

**Detailed Technical Report**  
**Duality AI Off‑Road Segmentation Challenge**

- **Team Name:** CrypticByte  
- **Date:** 26 February 2026  

---

## 1. Introduction

Semantic segmentation is a computer vision task in which each pixel of an image is assigned a semantic label. This capability is essential for autonomous systems that must understand their surroundings at a fine level of detail. In off‑road environments, accurate segmentation enables obstacle avoidance, terrain analysis, path planning, and safe navigation.

This project focuses on segmenting synthetic desert scenes generated using a digital twin platform. Unlike urban datasets, off‑road scenes contain irregular terrain, sparse structures, heavy texture variation, and extreme lighting changes, making the task particularly challenging.

---

## 2. Dataset Description

The dataset consists of synthetic RGB images paired with pixel‑wise ground truth segmentation masks. Each pixel belongs to one of 10 semantic classes representing natural objects and terrain components.

**Semantic classes:**

- Trees  
- Lush Bushes  
- Dry Grass  
- Dry Bushes  
- Ground Clutter  
- Flowers  
- Logs  
- Rocks  
- Landscape  
- Sky  

**Data splits:**

- **Training set** – used to learn model parameters  
- **Validation set** – used for model selection and tuning  
- **Test set** – unseen environment used to evaluate generalization  

The dataset exhibits strong class imbalance: large regions such as **Sky** and **Landscape** dominate images, while small objects like **Logs** or **Rocks** appear rarely.

---

## 3. Data Preprocessing and Augmentation

All images were resized to a fixed resolution to ensure consistent input size. Pixel values were normalized using ImageNet mean and standard deviation to keep training stable.

Segmentation masks originally contained raw pixel IDs. These were converted to contiguous class indices (0–9) to match the model output format.

To improve generalization, several data augmentation techniques were applied:

**Photometric augmentations:**

- Brightness adjustment  
- Contrast adjustment  

These simulate different lighting conditions encountered in off‑road scenes.

**Geometric transformations:**

- Random flips  
- Random crops or small scale changes  

These increase spatial diversity and reduce overfitting to specific viewpoints.

The final configuration primarily used brightness and contrast jitter because it improved robustness without destabilizing training.

---

## 4. Model Architecture

The final model selected is **U‑Net**, a convolutional encoder‑decoder architecture designed specifically for semantic segmentation.

- The **encoder** gradually reduces spatial resolution while extracting high‑level semantic features.  
- The **decoder** progressively restores resolution to produce dense pixel‑level predictions.  

A key feature of U‑Net is the use of **skip connections** that transfer fine‑grained spatial information from encoder layers directly to decoder layers. This enables precise boundary reconstruction and improves segmentation of small or thin structures.

The network outputs a multi‑channel feature map where each channel represents the predicted probability of a particular class at every pixel location.

---

## 5. Training Strategy and Hyperparameters

Training was performed using supervised learning with ground truth masks.

**Optimization:**

- Optimizer: **Adam** (adaptive learning, fast convergence)  
- Learning rate: **1e‑4** (found to provide stable training)  

**Loss functions:**

- **Weighted Cross‑Entropy Loss** – optimizes overall classification accuracy and compensates for class imbalance by up‑weighting rare classes.  
- **Dice Loss** – improves region overlap, especially for small or rare objects.  

The combined objective encourages both correct classification and good spatial overlap.

Batch size was limited by GPU memory constraints. After each epoch, validation performance was monitored, and the best model checkpoint was selected based on validation mean IoU.

---

## 6. Results and Performance

Multiple architectures were evaluated to determine the best model. Baseline approaches showed moderate accuracy but struggled with small objects or complex textures. After systematic tuning of data, architecture, and training hyperparameters, **U‑Net** achieved the best balance between accuracy and computational efficiency.

Performance was measured using standard segmentation metrics:

- Mean Intersection over Union (**Mean IoU**)  
- **Dice Score**  
- **Pixel Accuracy**  
- **Inference Time per image**

**Final measured values (validation set, 317 images):**

- **Mean IoU:** 0.6082  
- **Dice Score:** 0.7185  
- **Pixel Accuracy:** 0.8890  
- **Inference Time:** ≈ 2.2 ms/image  

These results show that the final model provides strong overlap quality and high pixel‑wise accuracy while remaining efficient enough for real‑time or near real‑time deployment.

---

## 7. Challenges and Solutions

Several technical challenges were encountered during development:

- **Class Imbalance** – Rare classes (e.g., Logs, Rocks, Flowers) were difficult to learn.  
  
  - *Solution:* Weighted losses and Dice optimization were used to emphasize minority classes.

- **Visual Similarity** – Some terrain types share similar textures and colors, causing confusion between classes.  
  
  - *Solution:* Data augmentation improved robustness and reduced misclassification.

- **Small Object Detection** – Thin or tiny objects were often missed in early experiments.  
  
  - *Solution:* U‑Net’s skip connections improved localization and boundary accuracy.

- **Generalization to New Environments** – Test images came from a different environment than the training set.  
  
  - *Solution:* Robust augmentations and careful regularization helped prevent overfitting to training scenes.

---

## 8. Conclusion

The final U‑Net model demonstrates strong performance for semantic segmentation of synthetic off‑road environments. It effectively balances:

- Accuracy on major terrain and sky classes  
- Robustness under varying illumination and viewpoints  
- Computational efficiency suitable for real‑time or near real‑time applications  

This makes it a promising candidate for integration into autonomous navigation pipelines in off‑road settings.

---

## 9. Future Work

Possible future improvements include:

- Scaling to larger and more diverse datasets.  
- Exploring advanced architectures (e.g., UNet++, DeepLabV3+, transformer‑based models).  
- Applying domain adaptation techniques to transfer from synthetic to real‑world off‑road data.  
- Improving rare‑class detection with class‑balanced sampling, focal losses, or specialized rare‑object detectors.

---

## 10. References and Acknowledgements

**References:**

- Ronneberger et al., *U‑Net: Convolutional Networks for Biomedical Image Segmentation*  
- PyTorch Documentation  
- Duality AI Off‑Road Segmentation Challenge materials  

**Acknowledgements:**  
We thank the organizers, mentors, and teammates for their support throughout this project.
