"""
Standalone Image Inference Script
=================================
Predicts cancer malignancy on any image file:
    python predict_standalone.py --image static/samples/malignant_melanoma_01.png
    python predict_standalone.py --image static/samples/benign_nevus_01.png --algorithm svm
"""

import os
import argparse
from pipeline import ImagePreprocessor, FeatureExtractor, CancerClassifier


def predict_image(image_path: str, algorithm: str = "random_forest"):
    print("=" * 65)
    print("       OncoVision AI - Standalone Image Cancer Inference        ")
    print("=" * 65)

    if not os.path.exists(image_path):
        print(f"[!] Error: Image not found at '{image_path}'")
        return

    print(f"[*] Loading Image: {image_path}")
    raw_img = ImagePreprocessor.file_to_cv2(image_path)
    print(f"    Dimensions: {raw_img.shape[1]}x{raw_img.shape[0]} px | Channels: {raw_img.shape[2]}")

    print("[*] Stage 2: Running Image Preprocessing & Lesion Segmentation...")
    prep = ImagePreprocessor.process_all_stages(raw_img)
    
    print("[*] Stage 3: Extracting GLCM, Color Moments & ABCD Morphological Features...")
    features = FeatureExtractor.extract_features(prep["data"])
    
    print("\n--- Key Quantitative Features ---")
    print(f"  * GLCM Contrast:         {features['glcm_contrast']:.3f}")
    print(f"  * GLCM Homogeneity:      {features['glcm_homogeneity']:.3f}")
    print(f"  * Asymmetry Index (A):   {features['asymmetry_index']:.3f}")
    print(f"  * Border Irregularity(B):{features['border_irregularity']:.3f}")
    print(f"  * Color Variegation (C): {features['color_variegation']:.3f}")
    print(f"  * Equivalent Diam. (D):  {features['equivalent_diameter']:.1f} px")

    print(f"\n[*] Stage 4 & 5: Running Machine Learning Inference ({algorithm.upper()})...")
    classifier = CancerClassifier(algorithm=algorithm)
    loaded = classifier.load_model()
    if not loaded:
        print("[!] Pre-trained model not found. Training baseline model now...")
        from generate_dataset import generate_training_dataset
        df = generate_training_dataset(300)
        X = df[FeatureExtractor.FEATURE_NAMES].values
        y = df["target"].values
        classifier.train(X, y)

    result = classifier.predict_sample(features)

    print("\n" + "=" * 65)
    print("                   DIAGNOSTIC VERDICT                    ")
    print("=" * 65)
    verdict_badge = "[ MALIGNANT ]" if result["is_malignant"] else "[ BENIGN ]"
    print(f"  VERDICT:                 {verdict_badge} ({result['prediction'].upper()})")
    print(f"  Confidence:              {result['confidence_percentage']}%")
    print(f"  Malignancy Probability:  {result['probability_malignant'] * 100:.2f}%")
    print(f"  Benign Probability:      {result['probability_benign'] * 100:.2f}%")
    print(f"  Risk Stratification:     {result['risk_tier']}")
    print(f"\n  Clinical Advice:\n  \"{result['clinical_recommendation']}\"")
    print("=" * 65)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Cancer Malignancy from Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input lesion image")
    parser.add_argument("--algorithm", type=str, default="random_forest", choices=["random_forest", "svm", "knn", "logistic_regression", "gradient_boosting"])
    args = parser.parse_args()

    predict_image(args.image, args.algorithm)
