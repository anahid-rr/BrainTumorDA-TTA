# BrainTumorDA-TTA

**MRI Brain Tumor Classification with CNN, Data Augmentation, and Planned Domain/Test-Time Adaptation**

---

## Project Overview

This project aims to classify brain MRI images into four categories: **glioma tumor, meningioma tumor, pituitary tumor, and no tumor**. The initial phase focuses on implementing a **baseline CNN classifier** and improving robustness with **data augmentation**.  

In later stages, **Domain Adaptation (DA)** and **Test-Time Adaptation (TTA)** techniques will be incorporated to improve model generalization across different MRI domains and acquisition variations, aligning with the original research proposal.  

The project uses a **supervised learning approach**, with labeled MRI images organized into class-based directories. The current dataset is sourced from Kaggle and contains all four tumor classes needed for type classification.  

---

## Repository Contents

- `data/` – Placeholder for dataset organization (train, validation, test folders)  
- `notebooks/` – Jupyter notebooks or Colab notebooks with training and evaluation code  
- `models/` – Saved models (optional)  
- `README.md` – This file  
- `requirements.txt` – Python dependencies  

---

## How to Run the Code

1. Clone the repository:  

```bash
git clone https://github.com/yourusername/BrainTumorDA-TTA.git
cd BrainTumorDA-TTA
```
2. Install required libraries:

```bash
pip install -r requirements.txt
```

3. Upload your dataset to Google Drive (or local directory) with the following structure:

```bash
ProjectData/
├─ BrainTumor/
│  ├─ Training/
│  │  ├─ glioma/
│  │  ├─ meningioma/
│  │  ├─ pituitary/
│  │  └─ no_tumor/
│  └─ Testing/
│     ├─ glioma/
│     ├─ meningioma/
│     ├─ pituitary/
│     └─ no_tumor/

```
4. Open the notebook notebooks/train_baseline.ipynb or notebooks/train_augmented.ipynb in Google Colab.

5. Update the dataset path variables (train_dir and test_dir) according to your setup.

6. Run the cells to train the baseline CNN and the augmented CNN.

## Dependencies / Libraries Used

- Python 3.8+

- TensorFlow 2.x

- NumPy

- Matplotlib

- scikit-learn

- You can install all dependencies using the included requirements.txt.

## Scripts / Modules Explanation

- train_baseline – Implements and trains a basic CNN classifier without augmentation.

- train_augmented – Trains the CNN with real-time data augmentation to improve generalization.

## Sample Data / Dataset Access

The project uses a Kaggle dataset for brain tumor classification containing four categories. If you do not include the data in the repository due to size restrictions, you can download it from:

Kaggle Brain Tumor Dataset

After downloading, organize it according to the folder structure shown above.

## Metrics & Visualization

- Training and validation accuracy/loss curves

- Confusion matrices for both baseline and augmented models

- Preliminary results indicate reduced overfitting with augmentation and improved validation performance
