"""
Standalone Model Training Script
================================
Run directly from terminal:
    python train_standalone.py --algorithm random_forest --split 0.2
    python train_standalone.py --algorithm svm --c 1.0 --kernel rbf
    python train_standalone.py --algorithm knn --k 5
"""

import os
import argparse
import pandas as pd
import numpy as np
from pipeline import FeatureExtractor, CancerClassifier


def run_training(algorithm: str, test_split: float, **kwargs):
    print("=" * 65)
    print("   OncoVision AI - Standalone Machine Learning Model Training   ")
    print("=" * 65)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cancer_features_dataset.csv")
    if not os.path.exists(data_path):
        print(f"[!] Dataset not found at {data_path}. Running generator first...")
        from generate_dataset import generate_training_dataset
        df = generate_training_dataset(300)
    else:
        df = pd.read_csv(data_path)

    print(f"[*] Dataset Loaded: {len(df)} total patient records")
    print(f"[*] Class Distribution: Benign={(df['target']==0).sum()} | Malignant={(df['target']==1).sum()}")
    print(f"[*] Selected Algorithm: {algorithm.upper()}")
    print(f"[*] Test Split Ratio: {test_split*100:.1f}%")

    features = FeatureExtractor.FEATURE_NAMES
    X = df[features].values
    y = df["target"].values

    classifier = CancerClassifier(algorithm=algorithm, **kwargs)
    metrics = classifier.train(X, y, test_size=test_split)

    print("\n" + "-" * 40)
    print("         PERFORMANCE EVALUATION         ")
    print("-" * 40)
    print(f"  Accuracy:                 {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision (Malignancy):   {metrics['precision']:.4f}")
    print(f"  Recall / Sensitivity:     {metrics['recall']:.4f}")
    print(f"  Specificity:              {metrics['specificity']:.4f}")
    print(f"  F1-Score:                 {metrics['f1_score']:.4f}")
    print(f"  Matthews Corr. Coeff:     {metrics['mcc']:.4f}")
    print(f"  ROC Area Under Curve:     {metrics['roc_auc']:.4f}")

    cm = metrics["confusion_matrix"]
    print("\n" + "-" * 40)
    print("            CONFUSION MATRIX            ")
    print("-" * 40)
    print(f"                  Predicted Benign  Predicted Malignant")
    print(f"  Actual Benign:        {cm['true_negative']:<18} {cm['false_positive']:<18} (TN / FP)")
    print(f"  Actual Malignant:     {cm['false_negative']:<18} {cm['true_positive']:<18} (FN / TP)")

    if metrics.get("feature_importances"):
        print("\n" + "-" * 40)
        print("        TOP 5 PREDICTIVE FEATURES       ")
        print("-" * 40)
        for i, item in enumerate(metrics["feature_importances"][:5], 1):
            print(f"  {i}. {item['name']:<24}: {item['importance']:.4f}")

    print("\n[OK] Model successfully trained and saved to 'models/' directory.")
    print("=" * 65)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cancer Prediction Classifier")
    parser.add_argument("--algorithm", type=str, default="random_forest", choices=["random_forest", "svm", "knn", "logistic_regression", "gradient_boosting"], help="Algorithm to train")
    parser.add_argument("--split", type=float, default=0.25, help="Test set fraction (e.g. 0.25)")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees (for RF / Gradient Boosting)")
    parser.add_argument("--c", type=float, default=1.0, help="Regularization C parameter for SVM / Logistic Regression")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for KNN")
    parser.add_argument("--kernel", type=str, default="rbf", help="SVM Kernel (linear, rbf, poly)")
    
    args = parser.parse_args()
    
    kwargs = {}
    if args.algorithm in ["random_forest", "gradient_boosting"]:
        kwargs["n_estimators"] = args.n_estimators
    if args.algorithm in ["svm", "logistic_regression"]:
        kwargs["C"] = args.c
    if args.algorithm == "svm":
        kwargs["kernel"] = args.kernel
    if args.algorithm == "knn":
        kwargs["n_neighbors"] = args.k

    run_training(args.algorithm, args.split, **kwargs)
