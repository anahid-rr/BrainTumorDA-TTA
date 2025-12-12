# Brain Tumor MRI Classification with Domain Shift Robustness

## Project Title
**Mitigating Domain Shift in Brain MRI: A Robust Approach to Tumor Type and Grade Prediction**

## Overview
This project addresses the critical challenge of domain shift in brain tumor classification from MRI scans. While deep learning models achieve high accuracy in controlled research settings, they often fail when deployed across different hospitals due to variations in imaging equipment and protocols. We develop and test robust classification systems that maintain performance despite these variations.

## Key Features
- Two-stage classification: tumor type detection (glioma, meningioma, pituitary, no tumor) and grade assessment (LGG vs HGG)
- Robust models tested under simulated domain shift conditions
- Comparison of custom CNNs vs transfer learning with VGG16
- Comprehensive evaluation framework for clinical deployment readiness

## Results Summary
| Model | Task | Clean Accuracy | Domain Shift Accuracy | Robustness |
|-------|------|----------------|----------------------|------------|
| 3-Layer CNN | Tumor Type | 94.9% | 88.1% | -6.79% |
| Fine-tuned VGG16 | Grade | 79.2% | 80.2% | +0.94% |

## Dataset Requirements

### Tumor Type Classification
- **Dataset**: Brain Tumor MRI Dataset (Kaggle)
- **Download Link**: [https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor
- **Total Images**: 7,023 JPG images
- **Format**: JPG (224×224 pixels)

### Grade Classification  
- **Dataset**: BraTS 2019 & 2020 (Brain Tumor Segmentation Challenge)
- **BraTS 2019**: [https://www.med.upenn.edu/cbica/brats2019/data.html](https://www.med.upenn.edu/cbica/brats2019/data.html)
- **BraTS 2020**: [https://www.med.upenn.edu/cbica/brats2020/data.html](https://www.med.upenn.edu/cbica/brats2020/data.html)
- **Classes**: Low-grade Glioma (LGG), High-grade Glioma (HGG)
- **Total Patients**: 704 (combined)
- **Original Format**: NIfTI (.nii.gz)
- **Required Format**: JPG

### Data Preparation

#### Converting NIfTI to JPG (for BraTS data)
Use the provided conversion script `convert_nii_to_jpg.ipynb` in the notebook to convert MRI scans from NIfTI format to JPG:
```python
# Example usage in notebook:
# Convert BraTS .nii files to JPG
python convert_nii_to_jpg.ipynb --input_dir /path/to/brats/data --output_dir /path/to/jpg/output
```

The conversion script:
- Extracts 2D slices from 3D NIfTI volumes
- Normalizes intensities to [0, 255]
- Saves as JPG images (224×224 pixels)
- Preserves patient IDs and labels

#### Directory Structure After Preparation
```
data/
├── tumor_type/
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── notumor/
│   └── Testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── pituitary/
│       └── notumor/
└── tumor_grade/
    ├── train/
    │   ├── LGG/
    │   └── HGG/
    └── test/
        ├── LGG/
        └── HGG/
```

## Installation

### Prerequisites
- Python 3.8+
- Google Colab with T4 GPU (recommended) or local GPU
- 8GB+ RAM

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/brain-tumor-domain-shift.git
cd brain-tumor-domain-shift

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
brain-tumor-domain-shift/
├── Mitigating_Domain_Shift_in_Brain_MRI(JPG).ipynb  # Main notebook with all code
├── README.md                         # This file
├── requirements.txt                  # Dependencies
├── data/                            # Place datasets here
│   ├── tumor_type/                  # Multi-class tumor dataset
│   └── tumor_grade/                 # BraTS grade classification

```

## Notebook Structure

The main notebook `Mitigating_Domain_Shift_in_Brain_MRI(JPG)` contains the following sections:

### 1. **Imports and Setup**
- Library imports
- GPU configuration
- Random seed setting

### 2. **Helper Functions**
- Data preprocessing utilities
- Visualization functions
- Metrics calculation

### 3. **Data Loading and Preprocessing**
- Dataset loading for both tasks
- Train/validation/test splitting
- Oversampling for class imbalance
- Image normalization (224×224, [0,1] range)

### 4. **Model Architectures**
- **Baseline CNN**: Simple 2-layer CNN
- **3-Layer CNN**: Enhanced CNN with deeper architecture
- **Baseline VGG16**: Frozen backbone with custom head
- **Fine-tuned VGG16**: Unfrozen last 4 layers

### 5. **Training Functions**
- Training loops with early stopping
- Learning rate scheduling
- Augmentation pipelines

### 6. **Experiments**
- **Experiment 1**: Tumor type classification without augmentation
- **Experiment 2**: Tumor type classification with augmentation
- **Experiment 3**: Grade classification without augmentation
- **Experiment 4**: Grade classification with domain shift

### 7. **Model Comparisons**
- Side-by-side confusion matrices
- ROC curves
- Performance metrics tables
- Training history plots

### 8. **Domain Shift Evaluation**
- Enhanced augmentation testing
- Robustness analysis
- Performance degradation measurement

## Usage

### Running in Google Colab
1. Upload the notebook to Google Colab
2. Mount your Google Drive containing the datasets:
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Update data paths in the notebook
4. Run all cells sequentially

### Running Locally
1. Open the notebook in Jupyter:
```bash
jupyter notebook Mitigating_Domain_Shift_in_Brain_MRI(JPG).ipynb
```
2. Update data paths to point to your local directories
3. Run cells sequentially

### Key Parameters to Modify
```python
# In the notebook, you can adjust these parameters:

# Data settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2

# Training settings
EPOCHS = 15  # for CNNs
EPOCHS_VGG = 10  # for VGG models
LEARNING_RATE = 1e-3  # for CNNs
LEARNING_RATE_FINETUNE = 1e-5  # for VGG fine-tuning

# Augmentation settings (modify for different domain shift levels)
augmentation_params = {
    'RandomFlip':"horizontal"
    'rotation': 0.1,
    'zoom': 0.1,
    'brightness': 0.4,
    'contrast': 0.3,
    'noise': 0.03
}
```

## Reproducing Results

1. **Prepare Data**: Organize datasets in the required folder structure
2. **Run Baseline**: Execute cells for baseline models without augmentation
3. **Run Enhanced Models**: Execute cells with augmentation enabled
4. **Domain Shift Testing**: Run the domain shift evaluation section
5. **Generate Visualizations**: Execute comparison cells for figures

## Key Findings
- Models with 92.6% clean accuracy can drop to 77.8% under domain shift
- 3-layer CNN maintains better robustness than baseline (88.1% vs 77.8%)
- Fine-tuning VGG16 improves grade classification from 64.2% to 79.2%
- Domain shift testing is essential for clinical deployment readiness


## Troubleshooting

**Common Issues:**

1. **GPU not detected**: Ensure CUDA and cuDNN are properly installed
2. **Memory errors**: Reduce batch size or image size
3. **Data path errors**: Update paths in notebook to match your directory structure
4. **Package conflicts**: Create a fresh virtual environment

## Future Work
- Extend to 3D volumetric analysis
- Implement uncertainty quantification
- Add attention mechanisms
- Develop medical-specific pretraining

## License

Licensed under the Apache License, Version 2.0.
