"""
Safe In-App Python Code Execution Sandbox
=========================================
Executes user-typed Python code in an isolated subprocess, captures stdout/stderr,
and returns execution telemetry.
"""

import sys
import subprocess
import tempfile
import time
import os

CODE_TEMPLATES = {
    "pipeline_demo": {
        "title": "Complete ML Pipeline Demo",
        "description": "Loads an image, preprocesses, extracts GLCM/ABCD features, and predicts malignancy.",
        "code": '''import os
from pipeline import ImagePreprocessor, FeatureExtractor, CancerClassifier

# 1. Load an image
image_path = "static/samples/malignant_melanoma_01.png"
print(f"Loading medical image from: {image_path}")
img = ImagePreprocessor.file_to_cv2(image_path)
print(f"Image shape: {img.shape} (Height, Width, Channels)")

# 2. Preprocess & Segment
print("\\nRunning preprocessing (Bilateral filter + CLAHE + Otsu segmentation)...")
prep = ImagePreprocessor.process_all_stages(img)
print("Preprocessing complete!")

# 3. Extract Features
print("\\nExtracting GLCM Texture & ABCD Morphological features...")
features = FeatureExtractor.extract_features(prep["data"])
print(f"GLCM Contrast:          {features['glcm_contrast']:.4f}")
print(f"GLCM Homogeneity:       {features['glcm_homogeneity']:.4f}")
print(f"Asymmetry Index:        {features['asymmetry_index']:.4f}")
print(f"Border Irregularity:    {features['border_irregularity']:.4f}")
print(f"Color Variegation:      {features['color_variegation']:.4f}")

# 4. Predict
print("\\nRunning inference with Random Forest classifier...")
clf = CancerClassifier(algorithm="random_forest")
clf.load_model()
verdict = clf.predict_sample(features)

print(f"\\n>>> RESULT: {verdict['prediction'].upper()} <<<")
print(f"Confidence: {verdict['confidence_percentage']}%")
print(f"Risk Tier:  {verdict['risk_tier']}")
print(f"Advice:     {verdict['clinical_recommendation']}")
'''
    },
    "model_comparison": {
        "title": "Benchmark ML Classifiers",
        "description": "Compares SVM, Random Forest, KNN, and Logistic Regression on the cancer dataset.",
        "code": '''import pandas as pd
from pipeline import FeatureExtractor, CancerClassifier

# Load training dataset
df = pd.read_csv("data/cancer_features_dataset.csv")
X = df[FeatureExtractor.FEATURE_NAMES].values
y = df["target"].values

algorithms = ["random_forest", "svm", "knn", "logistic_regression"]
print(f"{'Algorithm':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'ROC-AUC':<10}")
print("-" * 72)

for alg in algorithms:
    clf = CancerClassifier(algorithm=alg)
    metrics = clf.train(X, y, test_size=0.25)
    print(f"{alg.upper():<22} | {metrics['accuracy']*100:.1f}%     | {metrics['precision']:.4f}     | {metrics['recall']:.4f}     | {metrics['roc_auc']:.4f}")

print("\\nBenchmark complete! All models evaluated on test partition.")
'''
    },
    "glcm_deepdive": {
        "title": "GLCM & Texture Analysis",
        "description": "Deep-dives into Gray-Level Co-occurrence Matrix properties for lesion analysis.",
        "code": '''import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Load and convert image
img = cv2.imread("static/samples/benign_nevus_01.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Quantize to 16 gray levels
quantized = (gray // 16).astype(np.uint8)

# Compute GLCM at 4 directional angles (0, 45, 90, 135 deg)
distances = [1, 2]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
glcm = graycomatrix(quantized, distances=distances, angles=angles, levels=16, symmetric=True, normed=True)

print("--- GLCM Texture Matrix Properties ---")
for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
    val = graycoprops(glcm, prop)
    print(f"{prop.capitalize():<15}: Mean = {np.mean(val):.5f} | Directional spread = {np.std(val):.5f}")

print("\\nGLCM extraction successful. These metrics quantitatively describe surface irregularities.")
'''
    },
    "custom_feature": {
        "title": "Custom Feature Experimentation",
        "description": "Write and test your own mathematical cancer descriptor function.",
        "code": '''import cv2
import numpy as np

def calculate_lesion_entropy(image_path):
    """Computes Shannon Entropy of grayscale lesion pixel intensities."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256], density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

benign_path = "static/samples/benign_nevus_01.png"
malignant_path = "static/samples/malignant_melanoma_01.png"

e_benign = calculate_lesion_entropy(benign_path)
e_malignant = calculate_lesion_entropy(malignant_path)

print(f"Benign Sample Shannon Entropy:    {e_benign:.4f} bits/pixel")
print(f"Malignant Sample Shannon Entropy: {e_malignant:.4f} bits/pixel")
print(f"Entropy Difference:               {abs(e_malignant - e_benign):.4f}")
print("Higher entropy correlates with higher heterogeneity and tissue disorganization!")
'''
    }
}


def execute_user_code(code_str: str, timeout_sec: int = 15) -> dict:
    """Executes Python code in a safe subprocess and collects output."""
    start_time = time.time()
    
    # Write code to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        temp_path = f.name
        f.write(code_str)

    python_exe = sys.executable

    try:
        proc = subprocess.run(
            [python_exe, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        duration_ms = round((time.time() - start_time) * 1000, 2)
        
        return {
            "success": proc.returncode == 0,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "duration_ms": duration_ms
        }
    except subprocess.TimeoutExpired:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout_sec} seconds. Check for infinite loops.",
            "returncode": -1,
            "duration_ms": duration_ms
        }
    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "duration_ms": duration_ms
        }
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
