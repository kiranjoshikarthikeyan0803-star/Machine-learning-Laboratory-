# OncoVision AI — Cancer Prediction & Medical Machine Learning Studio

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E.svg)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0-5C3EE8.svg)](https://opencv.org/)
[![Status](https://img.shields.io/badge/Clinical%20Pipeline-Operational-10B981.svg)]()

> An end-to-end medical imaging platform integrating **Image Acquisition**, **Digital Preprocessing**, **Multi-Modal Feature Extraction (GLCM & ABCD)**, **Machine Learning Classifier Training**, **Clinical Performance Evaluation**, and an **In-App Interactive Python Code Studio** into a single unified workspace.

---

## 🌟 Key Highlights & Pipeline Stages

### Stage 1: Load Input Image
- **Dual Ingestion**: Drag-and-drop custom clinical dermoscopy/histopathology images (PNG, JPG, BMP) or select from 6 pre-validated benchmark samples (Melanocytic Nevi, Seborrheic Keratoses, Invasive Melanomas, Nodular Carcinomas).
- **Metadata Inspector**: Real-time extraction of spatial dimensions, color channels, aspect ratios, and luminance profiles.

### Stage 2: Computer Vision Preprocessing
- **Multi-Color Space Analysis**: Converts RGB specimens into Grayscale and HSV (Hue, Saturation, Value) representations to capture melanin variegation.
- **Denoising Engine**: Edge-preserving Bilateral Filter, Gaussian smoothing, or Median filtering to remove hair artifacts, skin noise, and reflection glare.
- **Adaptive Contrast Amplification**: Contrast Limited Adaptive Histogram Equalization (**CLAHE**) with configurable clip limit for localized tissue detail enhancement.
- **Automated Lesion Segmentation**: Otsu's adaptive binarization combined with morphological closing/opening to extract the exact lesion mask ROI and boundary contour.

### Stage 3: Multi-Modal Feature Extraction
Quantitative descriptors grounded in medical literature (ISIC, HAM10000) and dermatological diagnostic protocols:

1. **Gray-Level Co-occurrence Matrix (GLCM)**:
   - Quantizes textures across 4 directional angles (0°, 45°, 90°, 135°) at multiple spatial offsets.
   - Computes:
     - **Contrast**: $\sum_{i,j} |i - j|^2 \, p(i, j)$
     - **Dissimilarity**: $\sum_{i,j} |i - j| \, p(i, j)$
     - **Homogeneity**: $\sum_{i,j} \frac{p(i, j)}{1 + (i - j)^2}$
     - **Energy (ASM)**: $\sum_{i,j} p(i, j)^2$
     - **Correlation**: $\sum_{i,j} \frac{(i - \mu_i)(j - \mu_j) \, p(i, j)}{\sigma_i \sigma_j}$
2. **Color Moments**: Mean and standard deviation across R, G, B, and H, S, V channels, plus **Color Variegation Index**.
3. **ABCD Dermatology Criteria**:
   - **A (Asymmetry Index)**: Bi-axial flip residual difference normalized by lesion area:
     $$\text{Asymmetry} = \frac{\|M - M_{\text{flip}_h}\|_1 + \|M - M_{\text{flip}_v}\|_1}{2 \cdot \text{Area}}$$
   - **B (Border Irregularity)**: Isoperimetric quotient and compactness:
     $$\text{Compactness} = \frac{P^2}{4 \pi A}, \quad \text{Circularity} = \frac{4 \pi A}{P^2}$$
   - **C (Color Variegation)**: Multi-spectral spread across erythema, melanin, and slate zones.
   - **D (Equivalent Diameter)**: $D_{\text{equiv}} = \sqrt{\frac{4 A}{\pi}}$
4. **Sobel Edge Gradients**: Spatial edge magnitude distribution reflecting invasive tumor margins.
5. **Interactive Biometric Radar**: Dynamic SVG chart displaying the specimen's feature polygon compared against the clinical malignancy warning threshold.

### Stage 4: Machine Learning Classifier Arena
- **Supported Classifiers**:
  - **Random Forest Classifier**: Non-linear ensemble with Gini feature importance calculation.
  - **Support Vector Machine (SVM)**: RBF and Linear kernels for optimal hyper-plane separation.
  - **K-Nearest Neighbors (k-NN)**: Distance-weighted neighborhood classification.
  - **Logistic Regression**: Probabilistic log-odds modeling with L2 regularization.
  - **Gradient Boosting**: Sequential tree boosting.
- **Live Training & Tuning**: Configurable train/test split slider (15% to 40%), hyperparameter controls, and instant re-training on the 300-patient dataset.

### Stage 5: Performance Evaluation & Diagnostic Verdict
- **Clinical Diagnosis Card**: Instant `MALIGNANT` or `BENIGN` verdict, animated Confidence Percentage gauge, and clinical next-step recommendations.
- **Comprehensive Statistical Evaluation**:
  - Classification Accuracy
  - Precision (Positive Predictive Value)
  - Recall / Sensitivity (True Positive Rate)
  - Specificity (True Negative Rate)
  - F1-Score & Matthews Correlation Coefficient (MCC)
  - Receiver Operating Characteristic (**ROC AUC**)
- **Interactive Confusion Matrix Heatmap**: Displays True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
- **Interactive Vector ROC Curve**: Plots TPR against FPR with calculated AUC.
- **Printable Medical Report**: 1-click clinical export formatting patient specimen data.

### Stage 6: In-App Interactive Code Studio & Compiler
- **In-Browser Python Editor**: Edit, compile, and execute Python scripts directly in the web browser.
- **Pre-loaded Educational Templates**:
  - `pipeline_demo`: Complete pipeline execution from image loading to prediction.
  - `model_comparison`: Benchmark 4 machine learning models with side-by-side metrics.
  - `glcm_deepdive`: Extract and analyze GLCM texture matrices.
  - `custom_feature`: Write custom mathematical algorithms to inspect image entropy.
- **Integrated Terminal Console**: Live execution timer (ms), stdout stream, and error tracebacks.

---

## ⚡ Quick Launch Guide

### Option 1: 1-Click Launch (Windows)
Double-click `run.bat` in the project directory.

### Option 2: Command Line
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch web server
python app.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 🖥️ Standalone Command-Line Tools

### Model Training via CLI
```bash
python train_standalone.py --algorithm random_forest --split 0.25 --n_estimators 100
python train_standalone.py --algorithm svm --c 1.0 --kernel rbf
python train_standalone.py --algorithm knn --k 5
```

### Single-Image Diagnosis via CLI
```bash
python predict_standalone.py --image static/samples/malignant_melanoma_01.png
python predict_standalone.py --image static/samples/benign_nevus_01.png
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/status` | Returns active ML engine status and models |
| `GET` | `/api/samples` | Returns clinical benchmark sample list |
| `POST` | `/api/pipeline/process-sample` | Runs pipeline on a benchmark sample image |
| `POST` | `/api/pipeline/upload` | Uploads a custom file and executes pipeline |
| `POST` | `/api/train` | Re-trains selected classifier with hyperparameters |
| `POST` | `/api/run-code` | Compiles & runs Python code in the sandbox |
| `GET` | `/api/code-templates` | Retrieves preset educational code templates |

---

## 📄 License & Medical Disclaimer
This software is developed for research, education, and clinical decision-support prototyping. Always consult board-certified medical professionals for patient diagnostic decisions.
