"""
Cancer Prediction Machine Learning Pipeline
===========================================
Modular architecture implementing:
1. Image Loading & Verification
2. Image Preprocessing (Color spaces, Denoising, CLAHE, Otsu Segmentation)
3. Multi-modal Feature Extraction (GLCM Texture, Color Moments, ABCD Criteria, Edge Gradients)
4. Model Training & Validation (SVM, Random Forest, KNN, Logistic Regression, Gradient Boosting)
5. Performance Evaluation & Diagnostic Inference
"""

import io
import os
import base64
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, matthews_corrcoef
)
import joblib


class ImagePreprocessor:
    """Handles medical image transformations and visual step generation."""

    @staticmethod
    def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
        """Converts raw bytes to an RGB OpenCV image."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from bytes.")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def file_to_cv2(file_path: str) -> np.ndarray:
        """Loads an image file path into an RGB OpenCV image."""
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not load image at {file_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def cv2_to_base64(img_rgb: np.ndarray, format: str = "PNG") -> str:
        """Converts an RGB or Grayscale OpenCV image into a base64 data URI string."""
        if len(img_rgb.shape) == 2:
            # Grayscale
            pil_img = Image.fromarray(img_rgb)
        else:
            pil_img = Image.fromarray(img_rgb)
        buffered = io.BytesIO()
        pil_img.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"

    @classmethod
    def process_all_stages(
        cls,
        img_rgb: np.ndarray,
        filter_type: str = "bilateral",
        clahe_clip: float = 2.0,
        target_size: tuple = (256, 256)
    ) -> dict:
        """
        Executes all preprocessing steps and generates visual checkpoints for the UI.
        
        Returns:
            dict containing raw arrays and base64 preview representations.
        """
        # Resize to standard analysis resolution
        resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

        # 1. Grayscale Conversion
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

        # 2. HSV Color Space (valuable for melanoma melanin variegation)
        hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)

        # 3. Noise Filtering / Denoising
        if filter_type == "gaussian":
            denoised_gray = cv2.GaussianBlur(gray, (5, 5), 0)
            denoised_color = cv2.GaussianBlur(resized, (5, 5), 0)
        elif filter_type == "median":
            denoised_gray = cv2.medianBlur(gray, 5)
            denoised_color = cv2.medianBlur(resized, 5)
        else:  # Bilateral (preserves sharp lesion boundaries while smoothing skin noise)
            denoised_gray = cv2.bilateralFilter(gray, 9, 75, 75)
            denoised_color = cv2.bilateralFilter(resized, 9, 75, 75)

        # 4. Contrast Enhancement via CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(denoised_gray)

        # 5. Lesion Segmentation via Otsu's Adaptive Thresholding
        # Skin lesions are typically darker than surrounding tissue, so we invert or check mode
        _, thresh = cv2.threshold(clahe_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to remove artifacts and fill internal holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph_opened = cv2.morphologyEx(morph_closed, cv2.MORPH_OPEN, kernel, iterations=1)

        # Keep largest connected component (the primary lesion)
        contours, _ = cv2.findContours(morph_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(morph_opened)
        lesion_contour = None
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_cnt], -1, 255, -1)
            lesion_contour = largest_cnt
        else:
            mask = morph_opened

        # Overlay contour on original image for clinical inspection
        contour_overlay = resized.copy()
        if lesion_contour is not None:
            cv2.drawContours(contour_overlay, [lesion_contour], -1, (0, 255, 128), 2)
            # Draw minimum area bounding rect
            rect = cv2.minAreaRect(lesion_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(contour_overlay, [box], 0, (255, 60, 60), 1)

        # Segmented lesion ROI (masked original image)
        masked_lesion = cv2.bitwise_and(resized, resized, mask=mask)

        # Generate base64 data for all visual steps
        previews = {
            "original": cls.cv2_to_base64(resized),
            "grayscale": cls.cv2_to_base64(gray),
            "hsv_visual": cls.cv2_to_base64(hsv),
            "denoised": cls.cv2_to_base64(denoised_color),
            "clahe_enhanced": cls.cv2_to_base64(clahe_enhanced),
            "binary_mask": cls.cv2_to_base64(mask),
            "contour_overlay": cls.cv2_to_base64(contour_overlay),
            "segmented_roi": cls.cv2_to_base64(masked_lesion)
        }

        # Collect raw numpy data for feature extraction
        data = {
            "resized_rgb": resized,
            "gray": gray,
            "hsv": hsv,
            "clahe_enhanced": clahe_enhanced,
            "mask": mask,
            "lesion_contour": lesion_contour,
            "dimensions": {"width": img_rgb.shape[1], "height": img_rgb.shape[0], "channels": img_rgb.shape[2] if len(img_rgb.shape) == 3 else 1}
        }

        return {"previews": previews, "data": data}


class FeatureExtractor:
    """Extracts medical-grade statistical, texture (GLCM), color, and ABCD morphological features."""

    @classmethod
    def extract_features(cls, preprocessed_data: dict) -> dict:
        """
        Extracts all quantitative features required for cancer classification.
        
        Features Extracted:
        1. GLCM Texture (Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM)
        2. Color Moments (RGB and HSV mean, standard deviation, skewness)
        3. ABCD Dermatological Morphology (Asymmetry Index, Border Irregularity, Color Variegation, Diameter)
        4. Edge and Gradient Density (Sobel gradients)
        """
        gray = preprocessed_data["clahe_enhanced"]
        rgb = preprocessed_data["resized_rgb"]
        hsv = preprocessed_data["hsv"]
        mask = preprocessed_data["mask"]
        contour = preprocessed_data["lesion_contour"]

        # Ensure mask is binary (0 and 1)
        binary_mask = (mask > 127).astype(np.uint8)
        mask_pixels = binary_mask == 1
        num_lesion_pixels = np.sum(binary_mask)

        features = {}

        # -------------------------------------------------------------
        # 1. GLCM Texture Features (Gray-Level Co-occurrence Matrix)
        # -------------------------------------------------------------
        # Quantize grayscale to 16 levels to make GLCM computationally efficient and noise-robust
        quantized = (gray // 16).astype(np.uint8)
        # Compute GLCM for distances 1, 2 at angles 0, 45, 90, 135 degrees
        distances = [1, 2]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(quantized, distances=distances, angles=angles, levels=16, symmetric=True, normed=True)

        features["glcm_contrast"] = float(np.mean(graycoprops(glcm, "contrast")))
        features["glcm_dissimilarity"] = float(np.mean(graycoprops(glcm, "dissimilarity")))
        features["glcm_homogeneity"] = float(np.mean(graycoprops(glcm, "homogeneity")))
        features["glcm_energy"] = float(np.mean(graycoprops(glcm, "energy")))
        features["glcm_correlation"] = float(np.mean(graycoprops(glcm, "correlation")))
        features["glcm_asm"] = float(np.mean(graycoprops(glcm, "ASM")))

        # -------------------------------------------------------------
        # 2. Color Moments (RGB & HSV)
        # -------------------------------------------------------------
        if num_lesion_pixels > 0:
            r_vals = rgb[:, :, 0][mask_pixels]
            g_vals = rgb[:, :, 1][mask_pixels]
            b_vals = rgb[:, :, 2][mask_pixels]
            h_vals = hsv[:, :, 0][mask_pixels]
            s_vals = hsv[:, :, 1][mask_pixels]
            v_vals = hsv[:, :, 2][mask_pixels]
        else:
            r_vals, g_vals, b_vals = rgb[:, :, 0].flatten(), rgb[:, :, 1].flatten(), rgb[:, :, 2].flatten()
            h_vals, s_vals, v_vals = hsv[:, :, 0].flatten(), hsv[:, :, 1].flatten(), hsv[:, :, 2].flatten()

        features["mean_r"] = float(np.mean(r_vals))
        features["std_r"] = float(np.std(r_vals))
        features["mean_g"] = float(np.mean(g_vals))
        features["std_g"] = float(np.std(g_vals))
        features["mean_b"] = float(np.mean(b_vals))
        features["std_b"] = float(np.std(b_vals))
        features["mean_hue"] = float(np.mean(h_vals))
        features["std_hue"] = float(np.std(h_vals))
        features["mean_saturation"] = float(np.mean(s_vals))
        features["std_saturation"] = float(np.std(s_vals))

        # Color Variegation (standard deviation across color channels - indicator of malignancy)
        color_variances = [features["std_r"], features["std_g"], features["std_b"], features["std_hue"]]
        features["color_variegation"] = float(np.mean(color_variances))

        # -------------------------------------------------------------
        # 3. ABCD Dermatology Morphological Criteria
        # -------------------------------------------------------------
        # A - Asymmetry Index
        if num_lesion_pixels > 50:
            # Horizontal flip difference
            h_flipped = np.fliplr(binary_mask)
            diff_h = np.sum(np.abs(binary_mask.astype(float) - h_flipped.astype(float)))
            # Vertical flip difference
            v_flipped = np.flipud(binary_mask)
            diff_v = np.sum(np.abs(binary_mask.astype(float) - v_flipped.astype(float)))
            
            asymmetry_index = float((diff_h + diff_v) / (2.0 * max(num_lesion_pixels, 1)))
            asymmetry_index = min(max(asymmetry_index, 0.0), 1.0)
        else:
            asymmetry_index = 0.5
        features["asymmetry_index"] = asymmetry_index

        # B - Border Irregularity (Compactness / Circularity)
        if contour is not None and len(contour) >= 5:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            
            if area > 0:
                # Isoperimetric quotient / Compactness: 4 * pi * Area / Perimeter^2
                # Perfect circle = 1.0, irregular/notched lesions < 0.6
                circularity = float((4.0 * np.pi * area) / (perimeter ** 2 + 1e-6))
                compactness = float((perimeter ** 2) / (4.0 * np.pi * area + 1e-6))
            else:
                circularity = 0.5
                compactness = 1.5

            # Convex hull and solidity
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = float(area / (hull_area + 1e-6)) if hull_area > 0 else 0.5

            # Equivalent diameter
            equiv_diameter = float(np.sqrt(4.0 * area / (np.pi + 1e-6)))
        else:
            area = float(num_lesion_pixels)
            perimeter = 100.0
            circularity = 0.5
            compactness = 1.5
            solidity = 0.5
            equiv_diameter = float(np.sqrt(4.0 * max(area, 1) / np.pi))

        features["lesion_area"] = area
        features["perimeter"] = perimeter
        features["compactness"] = compactness
        features["circularity"] = circularity
        features["solidity"] = solidity
        features["border_irregularity"] = float(1.0 - min(circularity, 1.0))
        features["equivalent_diameter"] = equiv_diameter

        # -------------------------------------------------------------
        # 4. Edge and Gradient Density (Sobel)
        # -------------------------------------------------------------
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        features["edge_gradient_mean"] = float(np.mean(sobel_mag))
        features["edge_gradient_std"] = float(np.std(sobel_mag))

        return features

    @classmethod
    def get_feature_vector(cls, features_dict: dict, feature_order: list = None) -> np.ndarray:
        """Converts feature dictionary into standard ordered 1D numpy array."""
        if feature_order is None:
            feature_order = cls.FEATURE_NAMES
        vector = [features_dict.get(name, 0.0) for name in feature_order]
        return np.array(vector, dtype=np.float32)

    FEATURE_NAMES = [
        "glcm_contrast",
        "glcm_dissimilarity",
        "glcm_homogeneity",
        "glcm_energy",
        "glcm_correlation",
        "glcm_asm",
        "mean_r",
        "std_r",
        "mean_g",
        "std_g",
        "mean_b",
        "std_b",
        "mean_hue",
        "std_hue",
        "mean_saturation",
        "std_saturation",
        "color_variegation",
        "asymmetry_index",
        "border_irregularity",
        "compactness",
        "circularity",
        "solidity",
        "lesion_area",
        "equivalent_diameter",
        "edge_gradient_mean",
        "edge_gradient_std"
    ]


class CancerClassifier:
    """Manages training, hyperparameter configuration, evaluation, and inference."""

    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    def __init__(self, algorithm: str = "random_forest", **kwargs):
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = FeatureExtractor.FEATURE_NAMES
        self.is_trained = False
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        self._init_model()

    def _init_model(self):
        """Instantiates selected scikit-learn classifier with hyperparameters."""
        alg = self.algorithm.lower()
        if alg == "random_forest":
            n_estimators = self.kwargs.get("n_estimators", 100)
            max_depth = self.kwargs.get("max_depth", 10)
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                class_weight="balanced"
            )
        elif alg in ["svm", "svc"]:
            kernel = self.kwargs.get("kernel", "rbf")
            c = self.kwargs.get("C", 1.0)
            self.model = SVC(
                kernel=kernel,
                C=c,
                probability=True,
                random_state=42,
                class_weight="balanced"
            )
        elif alg == "knn":
            n_neighbors = self.kwargs.get("n_neighbors", 5)
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif alg == "logistic_regression":
            c = self.kwargs.get("C", 1.0)
            self.model = LogisticRegression(C=c, max_iter=1000, random_state=42, class_weight="balanced")
        elif alg == "gradient_boosting":
            n_estimators = self.kwargs.get("n_estimators", 100)
            lr = self.kwargs.get("learning_rate", 0.1)
            self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=lr, random_state=42)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.25, random_state: int = 42) -> dict:
        """
        Fits scaler and model, performs train-test evaluation, and computes performance metrics.
        
        Returns:
            dict containing comprehensive evaluation statistics.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Test set predictions
        y_pred = self.model.predict(X_test_scaled)
        if hasattr(self.model, "predict_proba"):
            y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        # Performance Metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = [int(val) for val in cm.ravel()]

        # Specificity
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # ROC Curve & AUC
        try:
            auc = float(roc_auc_score(y_test, y_proba))
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]
        except Exception:
            auc = 0.5
            roc_points = [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]

        # Feature Importance (if applicable)
        feature_importances = []
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            for name, imp in zip(self.feature_names, importances):
                feature_importances.append({"name": name, "importance": round(float(imp), 4)})
            feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)
        elif hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_[0])
            for name, coef in zip(self.feature_names, coefs):
                feature_importances.append({"name": name, "importance": round(float(coef), 4)})
            feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)

        results = {
            "algorithm": self.algorithm,
            "hyperparameters": self.kwargs,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "mcc": round(mcc, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp
            },
            "roc_curve": roc_points,
            "feature_importances": feature_importances[:10]
        }

        # Save model
        self.save_model()
        return results

    def predict_sample(self, features_dict: dict) -> dict:
        """Runs diagnostic prediction on single extracted feature dictionary."""
        if not self.is_trained:
            # Try to load existing model
            if not self.load_model():
                raise RuntimeError("Classifier has not been trained yet. Please train or load a model.")

        vector = FeatureExtractor.get_feature_vector(features_dict, self.feature_names)
        vector_scaled = self.scaler.transform([vector])

        pred_class = int(self.model.predict(vector_scaled)[0])
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(vector_scaled)[0]
            prob_benign = float(probs[0])
            prob_malignant = float(probs[1])
        else:
            prob_malignant = 1.0 if pred_class == 1 else 0.0
            prob_benign = 1.0 - prob_malignant

        is_malignant = pred_class == 1
        confidence = prob_malignant if is_malignant else prob_benign

        # Clinical Risk Stratification
        if is_malignant:
            risk_tier = "High Risk (Critical)" if prob_malignant >= 0.8 else "Moderate-to-High Risk"
            verdict = "Malignant"
            clinical_advice = (
                "Immediate dermatological/oncological biopsy and histological confirmation recommended. "
                "Asymmetry and border irregularities require clinical correlation."
            )
        else:
            if prob_benign >= 0.85:
                risk_tier = "Low Risk (Likely Benign)"
                clinical_advice = "Regular periodic monitoring advised. No acute malignancy features detected."
            else:
                risk_tier = "Moderate (Indeterminate - Follow-up)"
                clinical_advice = "Mild atypical features observed. Re-evaluation in 3 months recommended."
            verdict = "Benign"

        return {
            "prediction": verdict,
            "is_malignant": is_malignant,
            "confidence_percentage": round(confidence * 100, 2),
            "probability_malignant": round(prob_malignant, 4),
            "probability_benign": round(prob_benign, 4),
            "risk_tier": risk_tier,
            "clinical_recommendation": clinical_advice,
            "algorithm_used": self.algorithm
        }

    def save_model(self, filename: str = None):
        """Persists trained model and scaler to disk."""
        if filename is None:
            filename = f"{self.algorithm}_model.joblib"
        filepath = os.path.join(self.MODEL_DIR, filename)
        joblib.dump({"model": self.model, "scaler": self.scaler, "feature_names": self.feature_names}, filepath)

    def load_model(self, filename: str = None) -> bool:
        """Loads model and scaler from disk if available."""
        if filename is None:
            filename = f"{self.algorithm}_model.joblib"
        filepath = os.path.join(self.MODEL_DIR, filename)
        if os.path.exists(filepath):
            bundle = joblib.load(filepath)
            self.model = bundle["model"]
            self.scaler = bundle["scaler"]
            self.feature_names = bundle.get("feature_names", FeatureExtractor.FEATURE_NAMES)
            self.is_trained = True
            return True
        return False
