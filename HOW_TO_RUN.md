# OncoVision AI — How to Compile, Build, and Run the Project

Welcome to **OncoVision AI**, a complete machine learning computer vision system for **Cancer Prediction & Lesion Diagnosis**.

This system covers all 5 fundamental stages in a single unified frame:
1. **Load Input Image** (Custom upload or clinical benchmark specimens)
2. **Preprocessor** (Grayscale, HSV, Denoising filters, CLAHE contrast enhancement, Otsu lesion segmentation)
3. **Feature Extraction** (GLCM texture, Color moments, ABCD dermatological criteria, Sobel edge gradients)
4. **Model Training & Tuning** (Random Forest, SVM, k-NN, Logistic Regression, Gradient Boosting)
5. **Performance Evaluation & Diagnosis** (Accuracy, Sensitivity, Specificity, ROC AUC, Confusion Matrix, Malignancy verdict)
6. **In-App Interactive Code Studio & Compiler** (Where you can type, edit, compile, and execute Python code directly inside the web interface)

---

## 🚀 1. Quick Start (Windows 1-Click Launch)

Simply double-click the included batch script in the project directory:

```cmd
run.bat
```

Or run via PowerShell:

```powershell
.\run.ps1
```

This will automatically check your Python environment, load the benchmark images and dataset, start the web server at `http://localhost:8000`, and launch your default browser automatically.

---

## 📦 2. Manual Installation & Prerequisites

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Web browser (Chrome, Edge, Firefox, Brave)

### Step 1: Open Terminal / Command Prompt
Navigate to the project root folder:
```cmd
cd "c:\Users\Lenovo\OneDrive\Desktop\Cancer Prediction"
```

### Step 2: Install Required Python Packages
```cmd
pip install -r requirements.txt
```
*(All core packages: `numpy`, `scipy`, `scikit-learn`, `scikit-image`, `opencv-python`, `pillow`, `fastapi`, `uvicorn`, `pandas`, `joblib`)*

---

## 🌐 3. How to Run the Web Application

Launch the server using Python:

```cmd
python app.py
```

*If `python` is not in your system PATH on Windows, run:*
```powershell
& "$env:LOCALAPPDATA\Python\bin\python.exe" app.py
```

Once running, navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

### In the Web Application, You Can:
- **Load Images**: Click any of the 6 clinical benchmark cards (3 Benign, 3 Malignant) or drag-and-drop your own dermoscopy/histopathology image.
- **Inspect Transformations**: View all 5 image processing stages side-by-side (Original, Grayscale, Denoised, CLAHE, Otsu contour).
- **Tune Preprocessing**: Change filter types (Bilateral, Gaussian, Median) or adjust the CLAHE clip limit live.
- **Inspect Features**: View GLCM texture and ABCD clinical values, alongside an interactive SVG Biometric Radar Chart.
- **Train Models**: Switch between Random Forest, SVM, KNN, or Logistic Regression, adjust hyperparameters and test split, and train in real-time.
- **Review Evaluation**: Inspect the diagnosis verdict banner, confidence gauge, confusion matrix heatmap, and ROC curve.
- **Export Report**: Click **"Export Clinical Report"** to generate a printable summary.

---

## 💻 4. How to Type & Run Programs in the In-App Code Studio

You can type, edit, compile, and run Python code directly inside the web browser!

1. In the web interface, click the **"Code Sandbox"** button in the top navigation bar.
2. An interactive Python code editor and terminal console will open on the right side of the screen.
3. Select any pre-built template from the dropdown or type your own Python program:
   - **Template 1: Full Pipeline Demo**: Loads an image, segments it, extracts features, and outputs diagnosis.
   - **Template 2: Benchmark 4 Classifiers**: Compares Random Forest, SVM, KNN, and Logistic Regression on the dataset.
   - **Template 3: GLCM Texture Analysis**: Deep dive into the Gray-Level Co-occurrence Matrix properties.
   - **Template 4: Custom Feature Function**: Write custom mathematical algorithms to analyze cancer images.
4. Click **"Run Program" (▶)**.
5. Watch the live terminal console output stdout, execution time (in ms), return codes, and error traces!

---

## 🖥️ 5. Standalone CLI Programs (Command-Line Execution)

If you prefer executing programs directly from the Windows Command Prompt or PowerShell, the project includes standalone CLI tools:

### A. Train & Evaluate Machine Learning Models
```cmd
# 1. Train Random Forest (Default)
python train_standalone.py --algorithm random_forest --split 0.25 --n_estimators 100

# 2. Train Support Vector Machine (SVM with RBF Kernel)
python train_standalone.py --algorithm svm --c 1.0 --kernel rbf

# 3. Train K-Nearest Neighbors (k-NN)
python train_standalone.py --algorithm knn --k 5

# 4. Train Logistic Regression
python train_standalone.py --algorithm logistic_regression --c 1.0
```

**Output Displayed in Terminal:**
- Total training and testing samples
- Accuracy, Precision, Recall / Sensitivity, Specificity, F1-Score, MCC, ROC AUC
- Formatted 2×2 Confusion Matrix
- Top 5 Gini Predictive Features
- Model saved into `models/` directory

---

### B. Run Cancer Inference on Any Image via CLI
```cmd
# Predict on Malignant Melanoma Sample
python predict_standalone.py --image static/samples/malignant_melanoma_01.png

# Predict on Benign Nevus Sample
python predict_standalone.py --image static/samples/benign_nevus_01.png

# Predict using a specific algorithm
python predict_standalone.py --image static/samples/malignant_melanoma_01.png --algorithm svm
```

**Output Displayed in Terminal:**
- Image dimensions and channels
- Preprocessing & segmentation status
- Numerical values of GLCM Contrast, Homogeneity, Asymmetry Index, Border Irregularity, and Color Variegation
- **Final Verdict**: `[ MALIGNANT ]` or `[ BENIGN ]`
- Confidence percentage & class probabilities
- Clinical Risk Tier (High / Moderate / Low)
- Dermatological recommendations

---

### C. Re-generate Benchmark Samples & Clinical Dataset
```cmd
python generate_dataset.py
```
Synthesizes 6 dermoscopic benchmark image specimens and generates `data/cancer_features_dataset.csv` with 300 validated patient records.

---

## 📁 6. Project Architecture Overview

```
Cancer Prediction/
├── app.py                      # FastAPI Web Application & REST API
├── pipeline.py                 # Core ML Pipeline (Preprocessor, FeatureExtractor, Classifier)
├── generate_dataset.py         # Benchmark generator for images and 300-patient dataset
├── train_standalone.py         # Standalone CLI script for model training & evaluation
├── predict_standalone.py       # Standalone CLI script for single-image diagnosis
├── run_code_sandbox.py         # Isolated Python execution runner for in-app code editor
├── requirements.txt            # Python dependencies
├── run.bat                     # Windows 1-click launch batch script
├── run.ps1                     # PowerShell launch script
├── HOW_TO_RUN.md               # Step-by-step compilation & execution guide
├── README.md                   # Full theoretical and mathematical documentation
├── data/
│   └── cancer_features_dataset.csv # 300 patient records with ground-truth labels
├── models/
│   ├── random_forest_model.joblib  # Trained model checkpoint
│   ├── svm_model.joblib            # Trained model checkpoint
│   └── knn_model.joblib            # Trained model checkpoint
└── static/
    ├── index.html              # Modern single-frame web interface
    ├── css/
    │   └── style.css           # Premium cyber-medical dark theme & glassmorphism
    ├── js/
    │   └── app.js              # Live charts, interactive pipeline, and code runner logic
    └── samples/                # 6 clinical benchmark lesion images
```

---

## 🛠️ 7. Troubleshooting & FAQ

1. **Python was not found**:
   Ensure Python is installed. If using Windows Python from AppData, use:
   `& "$env:LOCALAPPDATA\Python\bin\python.exe" app.py`
2. **Port 8000 already in use**:
   Change the port by setting the `PORT` environment variable:
   `set PORT=8080 && python app.py`
3. **Can I use my own medical images?**
   Yes! Drag and drop any `.png`, `.jpg`, or `.jpeg` dermoscopy or histology image directly onto the upload zone in the web interface, or run:
   `python predict_standalone.py --image "path\to\your_image.jpg"`
