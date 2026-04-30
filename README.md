# Explainable AI for Pulmonary Nodule Classification

This repository contains a deep learning pipeline for **pulmonary nodule classification** on 3D CT scans using the **LUNA16 dataset**.  
The project focuses not only on classification performance, but also on **explainability**, by comparing multiple XAI methods applied to a 3D CNN model.

> **Academic/research project only.** This system is not intended for real clinical diagnosis.

---

## Project Overview

The objective of this project is to classify lung nodules from 3D CT scan patches and evaluate how well explainability methods highlight the regions that influence the model decision.

The pipeline includes:

- CT scan loading and preprocessing
- 3D patch extraction around candidate nodules
- Patient-level K-Fold cross-validation
- 3D deep learning model training
- Performance evaluation using medical imaging metrics
- Explainable AI visualization and quantitative comparison
- Model card and clinical validation material generation

---

## Dataset

This project uses the **LUNA16 dataset**.

Expected files include:

- `annotations.csv`
- `candidates_V2.csv`
- CT scans in `.mhd` format
- Optional lung segmentation masks

The dataset is not included in this repository because of its size.  
You must download it separately and update the paths in the notebook configuration if needed.

Default Kaggle-style paths used in the notebook:

```python
root = "/kaggle/input/datasets/avc0706/luna16"
annotations_csv = "/kaggle/input/datasets/avc0706/luna16/annotations.csv"
candidates_csv = "/kaggle/input/datasets/avc0706/luna16/candidates_V2/candidates_V2.csv"
```

---

## Main Technologies

- Python
- PyTorch
- SimpleITK
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- SHAP
- DiskCache
- tqdm

---

## Model Architecture

The main configured model is:

- **3D ResNet-18**

The code also includes support for:

- 3D ResNet-34
- 3D ResNet-50
- 3D U-Net classifier

The model processes 3D CT patches and predicts whether a candidate region corresponds to a pulmonary nodule.

---

## Evaluation Metrics

The project evaluates classification performance using several metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Matthews Correlation Coefficient
- Cohen’s Kappa
- Confusion Matrix
- FROC
- CPM
- Calibration analysis
- Bootstrap confidence intervals

---

## Explainable AI Methods

The following XAI methods are implemented and compared:

- **Grad-CAM 3D**
- **Integrated Gradients**
- **Occlusion Sensitivity**

The explanations are evaluated using quantitative metrics such as:

- Insertion/Deletion AUC
- Pointing Game
- Attribution IoU
- Energy-based Pointing Game
- Explanation Consistency
- Sparsity

---

## Repository Structure

```text
.
├── pfa-reportversion.ipynb     # Main notebook containing the full pipeline
├── README.md                   # Project documentation
├── saved_models/               # Generated trained model checkpoints
└── results/                    # Generated plots, metrics, XAI outputs, and reports
```

Generated outputs may include:

```text
results/
├── eval_fold*.png
├── kfold_summary.json
├── xai_comparison.png
├── xai_summary.csv
├── xai_vis_sample*.png
├── clinical_validation/
└── MODEL_CARD.txt
```

---

## Installation

Install the required dependencies:

```bash
pip install torch numpy pandas scipy scikit-learn SimpleITK matplotlib shap tqdm diskcache
```

If you are using Jupyter:

```bash
pip install notebook
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

2. Install the dependencies:

```bash
pip install torch numpy pandas scipy scikit-learn SimpleITK matplotlib shap tqdm diskcache
```

3. Open the notebook:

```bash
jupyter notebook pfa-reportversion.ipynb
```

4. Update the dataset paths in the configuration section if you are not using the default Kaggle paths.

5. Run the notebook cells.

---

## Main Pipeline

The notebook follows these main steps:

1. Load LUNA16 annotations and candidate nodules
2. Load CT scans and extract 3D patches
3. Apply preprocessing and data augmentation
4. Train a 3D CNN using patient-level K-Fold cross-validation
5. Evaluate the model using classification and medical imaging metrics
6. Generate XAI explanations
7. Compare XAI methods quantitatively
8. Save results, plots, model checkpoints, and the model card

---

## Results

The notebook automatically saves the following outputs:

- Trained model checkpoint
- K-Fold evaluation summary
- Evaluation plots
- XAI comparison plots
- XAI visualizations on CT slices
- Clinical validation protocol
- Model card

---

## Notes

- Patient-level splitting is used to reduce data leakage between training and validation folds.
- The project is designed for GPU execution because 3D CT processing and 3D CNN training are computationally expensive.
- Dataset files are not included in this repository.
- The results may vary depending on hardware, random seed, dataset version, and training configuration.

---

## Author

**Yessine Kammoun**  
Telecommunications Engineering Student  
ENET'COM Sfax
