# Workflow CI/CD MLflow - Muhammad Rivaldy Pratama

Repository untuk submission Dicoding - Membangun Sistem Machine Learning (Kriteria 3 - ADVANCE)

## Struktur Folder

```
Workflow-CI-Muhammad-Rivaldy-Pratama/
├── .github/workflows/
│   └── mlproject_ci.yml           # GitHub Actions untuk CI/CD
└── MLProject/
    ├── MLProject                  # MLflow project config
    ├── conda.yaml                 # Environment dependencies
    ├── modelling_advanced.py      # Training script
    └── data/preprocessed/
        ├── train_data.csv         # Data training
        ├── test_data.csv          # Data testing
        └── scaler.pkl             # Scaler object
```

## Kriteria yang Dipenuhi

✅ **Kriteria 3: ADVANCE (4 pts)**
- MLflow Project structure lengkap
- GitHub Actions CI/CD workflow
- Artefak tersimpan ke repository
- Docker image build & push to Docker Hub (optional)

## Cara Menjalankan

### 1. Local MLflow Run
```bash
mlflow run MLProject --env-manager=conda
```

### 2. GitHub Actions (Otomatis)
- Push ke branch main
- Workflow akan otomatis:
  - Setup environment
  - Train model dengan MLflow
  - Upload artifacts ke GitHub

### 3. Docker (Optional)
```bash
mlflow models build-docker -m runs:/<RUN_ID>/model -n rivaldy25lval/wine-quality:latest
docker push rivaldy25lval/wine-quality:latest
```

## Model Details

- Algorithm: Random Forest Classifier
- Hyperparameters: GridSearchCV optimized
- Metrics: Accuracy, Precision, Recall, F1, ROC AUC
- MLflow Tracking: DagsHub integration

## Author

Muhammad Rivaldy Pratama
- GitHub: [@Rivaldy-25-Lval](https://github.com/Rivaldy-25-Lval)
- DagsHub: [@Rivaldy-25-Lval](https://dagshub.com/Rivaldy-25-Lval)
- Dicoding: Rivaldy-25-Lval

## License

Submission project untuk Dicoding - Membangun Sistem Machine Learning
