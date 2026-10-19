# Machine Learning Laboratory

Welcome to the **Machine Learning Laboratory** repository. This repository houses all laboratory practical experiments (Experiments 1 through 14) and the final course project (**Cancer Prediction & Computer Vision ML Studio**).

---

## 📂 Repository Organization

```
Machine-learning-Laboratory-/
│
├── 📑 Laboratory Experiments (1 to 14)
│   ├── exp 1.ipynb             # Experiment 1
│   ├── exp 2.ipynb             # Experiment 2
│   ├── Exp 3.ipynb             # Experiment 3
│   ├── Exp4.pdf                # Experiment 4
│   ├── exp5.ipynb              # Experiment 5
│   ├── exp6.ipynb              # Experiment 6
│   ├── exp7.ipynb              # Experiment 7
│   ├── exp8.ipynb              # Experiment 8
│   ├── exp9.ipynb              # Experiment 9
│   ├── Exp11.pdf               # Experiment 11
│   ├── Exp12.pdf               # Experiment 12
│   ├── Exp13.ipynb             # Experiment 13
│   └── Exp14.ipynb             # Experiment 14
│
└── 🔬 Laboratory Final Project: Cancer Prediction
    └── Cancer Prediction/      # Complete Full-Pipeline Web Application
        ├── app.py              # FastAPI Web Application & REST Server
        ├── pipeline.py         # 5-Stage Machine Learning Pipeline Core
        ├── generate_dataset.py # Clinical benchmark generator & dataset synthesizer
        ├── train_standalone.py # Standalone model training CLI
        ├── predict_standalone.py # Standalone single-image prediction CLI
        ├── run_code_sandbox.py # In-browser Python sandbox executor
        ├── requirements.txt    # Python package dependencies
        ├── run.bat             # 1-Click Windows Batch Launcher
        ├── run.ps1             # PowerShell Launcher
        ├── HOW_TO_RUN.md       # Compilation & execution instructions
        ├── README.md           # Project architecture & mathematical documentation
        ├── data/               # 300 validated clinical patient records
        ├── models/             # Trained ML classifiers (Random Forest, SVM, k-NN)
        └── static/             # Frontend UI, CSS design system, and benchmark images
```

---

## 🚀 Quick Launch for Cancer Prediction Project

### 1-Click Launch (Windows)
Double-click `run.bat` (in the root or inside `Cancer Prediction/`):
```cmd
run.bat
```
This automatically verifies dependencies, starts the FastAPI server, and launches the web interface at **`http://localhost:8000`**.

### Detailed Project Documentation
For complete instructions on preprocessing, GLCM feature extraction, model training, and the interactive in-app code editor, see:
- 📖 [Cancer Prediction Project README](Cancer%20Prediction/README.md)
- 🛠️ [How to Compile & Run Guide](Cancer%20Prediction/HOW_TO_RUN.md)

---

## 🔒 Policy & External Contributions

Please review [CONTRIBUTING.md](CONTRIBUTING.md) and [LICENSE](LICENSE).  
This is a personal and academic laboratory repository. **No external pull requests, issues, or contributions are accepted.**
