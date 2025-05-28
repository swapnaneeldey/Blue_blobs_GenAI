# 🌌 Blue Blobs transfer learning model

A deep learning pipeline to identify **rare, low-mass, star-forming regions**—known as *blue blobs*—in the Virgo Cluster. This project combines **transfer learning** and **convolutional neural networks (CNNs)** to distinguish these elusive objects from typical galaxy backgrounds.

---

## 🚀 Overview

Blue blobs are isolated, low-stellar-mass systems that were previously unknown in large numbers. With only ~40 credible detections out of over 50,000 survey images, manual classification is highly inefficient. This repository uses **deep learning** to build a scalable classifier that can help filter images for follow-up.

---

## 📊 The Challenge

- ⚠️ **Tiny dataset**: Only 40 confirmed blue blob images
- ⚖️ **Class imbalance**: Abundant background/galaxy images
- ❓ **Rare object detection**: Conventional models struggle

---

## 🧠 Transfer Learning Approach

We used **pre-trained CNNs** (ImageNet weights) and fine-tuned them on our augmented dataset:

### Models Compared
- **ResNet50V2** (25.6M params)
- **ResNet101V2** (44.6M params)
- **EfficientNetB3** (11M params)

### Training Strategy
- **Feature Extraction**: Freeze backbone, train new dense head
- **Fine-Tuning**: Unfreeze last conv blocks and jointly train
- **Loss**: Binary cross-entropy  
- **Metrics**: Accuracy, AUC, Precision, Recall (per class)

---

## 🧪 Data Augmentation

To overcome the data bottleneck:
- Random flips, rotations, and zooms
- Rotated background sky behind the masked blue blob
- Downsampling for balanced class size

Final training data:
- ✅ 1255 blue blob images (augmented)
- ✅ 1340 background/galaxy images

---

## 🏆 Best Performing Model

**ResNet50V2 with ReLU dense head**:
- Two-layer head: 128-unit ReLU → 1-unit sigmoid
- Fine-tuned last block
- Best validation performance with minimal misclassification of blue blobs

---

## 🔭 Use Case

We recommend using the model not for final classification but as a **pre-filtering tool** in citizen science pipelines, reducing volunteer burden by filtering out clearly non-blue-blob candidates.

---

## 📁 Repository Structure

Blue_blobs_GenAI/
├── data/ # Augmented images
├── models/ # Trained models (.keras via Git LFS)
├── notebooks/ # Jupyter notebooks for training/evaluation
├── output/ # Metrics, plots, and saved histories
├── scripts/ # Script-based training interface
├── README.md # Project overview
└── requirements.txt # Python dependencies



---

## 🛠 Tech Stack

- Python 3.x
- TensorFlow / Keras
- Jupyter Notebooks
- Git LFS (for large model files)

---

## 🤝 Acknowledgments

- Guidance: Misha, Robert, and Arvind
- Citizen Scientists: >1600 volunteers on [Zooniverse](https://www.zooniverse.org/projects/mike-dot-jones-dot-astro/blobs-and-blurs-extreme-galaxies-in-clusters)

---

## 🌟 Future Work

- Improved domain-specific augmentations
- Use Grad-CAM for interpretability
- Deploy as a frontend interface for citizen science filtering

---

⭐ If you’re an employer or researcher interested in rare object detection, efficient modeling, or astronomy ML pipelines—feel free to explore or reach out!

