"""
OncoVision AI - Cancer Prediction & Machine Learning Pipeline Server
====================================================================
FastAPI Application serving REST API and unified single-frame interface.
"""

import os
import io
import json
from typing import Optional
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

from pipeline import ImagePreprocessor, FeatureExtractor, CancerClassifier
from run_code_sandbox import execute_user_code, CODE_TEMPLATES

app = FastAPI(
    title="OncoVision AI - Cancer Prediction System",
    description="Full-pipeline machine learning application for cancer detection from medical imaging.",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Default active classifier
active_classifier_name = "random_forest"
cached_classifiers = {}

def get_or_load_classifier(algorithm: str) -> CancerClassifier:
    global cached_classifiers
    if algorithm not in cached_classifiers:
        clf = CancerClassifier(algorithm=algorithm)
        if not clf.load_model():
            # If not yet trained, train on dataset
            dataset_path = os.path.join(DATA_DIR, "cancer_features_dataset.csv")
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path)
                X = df[FeatureExtractor.FEATURE_NAMES].values
                y = df["target"].values
                clf.train(X, y)
        cached_classifiers[algorithm] = clf
    return cached_classifiers[algorithm]


@app.get("/")
def serve_index():
    """Serves the main single-frame interface."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "OncoVision AI Server running. index.html not yet built."}


@app.get("/api/status")
def get_system_status():
    """Returns system status, active models, and available resources."""
    dataset_path = os.path.join(DATA_DIR, "cancer_features_dataset.csv")
    has_dataset = os.path.exists(dataset_path)
    num_samples = 0
    if has_dataset:
        try:
            df = pd.read_csv(dataset_path)
            num_samples = len(df)
        except Exception:
            pass

    available_models = ["random_forest", "svm", "knn", "logistic_regression", "gradient_boosting"]
    saved_models = []
    for m in available_models:
        if os.path.exists(os.path.join(MODELS_DIR, f"{m}_model.joblib")):
            saved_models.append(m)

    return {
        "status": "online",
        "system_name": "OncoVision AI Cancer Prediction Studio",
        "active_algorithm": active_classifier_name,
        "supported_algorithms": available_models,
        "saved_models": saved_models,
        "dataset_ready": has_dataset,
        "total_dataset_records": num_samples,
        "feature_count": len(FeatureExtractor.FEATURE_NAMES)
    }


@app.get("/api/samples")
def list_benchmark_samples():
    """Returns list of benchmark clinical samples available for 1-click testing."""
    samples = [
        {
            "id": "benign_nevus_01.png",
            "title": "Melanocytic Nevus",
            "category": "Benign",
            "is_malignant": False,
            "url": "/static/samples/benign_nevus_01.png",
            "description": "Regular borders, homogeneous pigmentation, high circularity."
        },
        {
            "id": "benign_keratosis_02.png",
            "title": "Seborrheic Keratosis",
            "category": "Benign",
            "is_malignant": False,
            "url": "/static/samples/benign_keratosis_02.png",
            "description": "Well-demarcated oval border, uniform melanin distribution."
        },
        {
            "id": "benign_dysplastic_03.png",
            "title": "Atypical Dysplastic Nevus",
            "category": "Benign",
            "is_malignant": False,
            "url": "/static/samples/benign_dysplastic_03.png",
            "description": "Mild border contouring, benign micro-architecture."
        },
        {
            "id": "malignant_melanoma_01.png",
            "title": "Invasive Melanoma",
            "category": "Malignant",
            "is_malignant": True,
            "url": "/static/samples/malignant_melanoma_01.png",
            "description": "Notched asymmetrical border, multi-chromatic pigment network."
        },
        {
            "id": "malignant_carcinoma_02.png",
            "title": "Nodular Carcinoma",
            "category": "Malignant",
            "is_malignant": True,
            "url": "/static/samples/malignant_carcinoma_02.png",
            "description": "Atypical pigmentation, high GLCM contrast, irregular perimeter."
        },
        {
            "id": "malignant_melanoma_03.png",
            "title": "Spreading Melanoma",
            "category": "Malignant",
            "is_malignant": True,
            "url": "/static/samples/malignant_melanoma_03.png",
            "description": "Severe asymmetry, multi-hued erythema and dark melanin zones."
        }
    ]
    return {"samples": samples}


class ProcessSampleRequest(BaseModel):
    sample_id: str
    filter_type: Optional[str] = "bilateral"
    clahe_clip: Optional[float] = 2.0
    algorithm: Optional[str] = "random_forest"


@app.post("/api/pipeline/process-sample")
def process_benchmark_sample(req: ProcessSampleRequest):
    """Processes one of the bundled clinical benchmark samples."""
    img_path = os.path.join(SAMPLES_DIR, req.sample_id)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail=f"Sample '{req.sample_id}' not found.")

    try:
        raw_img = ImagePreprocessor.file_to_cv2(img_path)
        return _run_pipeline_on_image(
            raw_img,
            filter_type=req.filter_type or "bilateral",
            clahe_clip=req.clahe_clip or 2.0,
            algorithm=req.algorithm or "random_forest",
            source_name=req.sample_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")


@app.post("/api/pipeline/upload")
async def process_uploaded_image(
    file: UploadFile = File(...),
    filter_type: str = Form("bilateral"),
    clahe_clip: float = Form(2.0),
    algorithm: str = Form("random_forest")
):
    """Uploads a user image and runs the full pipeline."""
    try:
        contents = await file.read()
        raw_img = ImagePreprocessor.bytes_to_cv2(contents)
        return _run_pipeline_on_image(
            raw_img,
            filter_type=filter_type,
            clahe_clip=clahe_clip,
            algorithm=algorithm,
            source_name=file.filename or "Uploaded Image"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")


def _run_pipeline_on_image(raw_img: np.ndarray, filter_type: str, clahe_clip: float, algorithm: str, source_name: str):
    """Executes Stages 2, 3, 4, 5 on raw image and formats output."""
    # Stage 2: Preprocess
    prep_result = ImagePreprocessor.process_all_stages(
        raw_img,
        filter_type=filter_type,
        clahe_clip=clahe_clip
    )

    # Stage 3: Feature Extraction
    features = FeatureExtractor.extract_features(prep_result["data"])

    # Stage 4 & 5: Prediction & Inference
    clf = get_or_load_classifier(algorithm)
    prediction = clf.predict_sample(features)

    # Radar chart / Fingerprint normalized values (0 to 1 scale for UI)
    radar_metrics = [
        {"axis": "Asymmetry (A)", "value": round(float(features["asymmetry_index"]), 3), "malignant_threshold": 0.45},
        {"axis": "Border Irregularity (B)", "value": round(float(features["border_irregularity"]), 3), "malignant_threshold": 0.40},
        {"axis": "Color Variegation (C)", "value": round(min(float(features["color_variegation"]) / 40.0, 1.0), 3), "malignant_threshold": 0.50},
        {"axis": "GLCM Contrast", "value": round(min(float(features["glcm_contrast"]) / 40.0, 1.0), 3), "malignant_threshold": 0.45},
        {"axis": "Edge Density", "value": round(min(float(features["edge_gradient_mean"]) / 60.0, 1.0), 3), "malignant_threshold": 0.50},
        {"axis": "Texture Entropy", "value": round(1.0 - float(features["glcm_homogeneity"]), 3), "malignant_threshold": 0.45}
    ]

    return {
        "source": source_name,
        "image_metadata": prep_result["data"]["dimensions"],
        "previews": prep_result["previews"],
        "features": {k: round(v, 4) if isinstance(v, (float, np.floating)) else v for k, v in features.items()},
        "radar_metrics": radar_metrics,
        "prediction": prediction
    }


class TrainRequest(BaseModel):
    algorithm: str = "random_forest"
    test_split: float = 0.25
    n_estimators: Optional[int] = 100
    c_param: Optional[float] = 1.0
    kernel: Optional[str] = "rbf"
    k_neighbors: Optional[int] = 5


@app.post("/api/train")
def train_model_endpoint(req: TrainRequest):
    """Re-trains selected classifier with hyperparameters and returns full evaluation."""
    dataset_path = os.path.join(DATA_DIR, "cancer_features_dataset.csv")
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Please generate dataset first.")

    df = pd.read_csv(dataset_path)
    X = df[FeatureExtractor.FEATURE_NAMES].values
    y = df["target"].values

    kwargs = {}
    if req.algorithm in ["random_forest", "gradient_boosting"]:
        kwargs["n_estimators"] = req.n_estimators or 100
    if req.algorithm in ["svm", "logistic_regression"]:
        kwargs["C"] = req.c_param or 1.0
    if req.algorithm == "svm":
        kwargs["kernel"] = req.kernel or "rbf"
    if req.algorithm == "knn":
        kwargs["n_neighbors"] = req.k_neighbors or 5

    clf = CancerClassifier(algorithm=req.algorithm, **kwargs)
    metrics = clf.train(X, y, test_size=req.test_split)

    # Cache as active model
    cached_classifiers[req.algorithm] = clf
    global active_classifier_name
    active_classifier_name = req.algorithm

    return {"message": "Model trained successfully", "metrics": metrics}


class CodeRunRequest(BaseModel):
    code: str
    timeout: Optional[int] = 15


@app.post("/api/run-code")
def run_code_endpoint(req: CodeRunRequest):
    """Executes user-typed Python code in the sandbox."""
    if not req.code or not req.code.strip():
        raise HTTPException(status_code=400, detail="Empty code submitted.")
    result = execute_user_code(req.code, timeout_sec=req.timeout or 15)
    return result


@app.get("/api/code-templates")
def get_code_templates():
    """Returns available Python code templates."""
    return {"templates": CODE_TEMPLATES}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("=" * 60)
    print(f"  Starting OncoVision AI Studio on http://localhost:{port}")
    print("=" * 60)
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
