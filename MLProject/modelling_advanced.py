import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, log_loss, matthews_corrcoef, cohen_kappa_score
)
import joblib
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(data_dir='data/preprocessed'):
    print(f"Loading data from: {data_dir}")
    
    train_data = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    
    # Heart Disease: target column is binary (0=No Disease, 1=Disease)
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    print(f"✅ Data loaded! Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]} (Heart Disease - 13 medical features)")
    print(f"   Target distribution - Train: {y_train.value_counts().to_dict()}")
    print(f"   Target distribution - Test: {y_test.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


def perform_hyperparameter_tuning(X_train, y_train):
    print("\nPerforming hyperparameter tuning...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best CV Score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all metrics for binary classification"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba[:, 1]),
        'log_loss': log_loss(y_true, y_pred_proba),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
    }
    
    # Additional metrics for binary classification
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['balanced_accuracy'] = (metrics['specificity'] + metrics['sensitivity']) / 2
    
    return metrics


def create_artifacts(y_true, y_pred, model, feature_names):
    """Create and save artifacts for binary classification"""
    os.makedirs('artifacts', exist_ok=True)
    
    # Confusion matrix (2x2 for binary classification)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title('Confusion Matrix - Heart Disease Classification', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('artifacts/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance (13 medical features)
    plt.figure(figsize=(12, 8))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices], color='skyblue')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.title('Feature Importances - Heart Disease Prediction', fontweight='bold')
    plt.ylabel('Importance')
    plt.xlabel('Medical Features')
    plt.tight_layout()
    plt.savefig('artifacts/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Classification report
    report = classification_report(y_true, y_pred, 
                                  target_names=['No Disease', 'Disease'],
                                  output_dict=True)
    with open('artifacts/classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    return ['artifacts/confusion_matrix.png', 
            'artifacts/feature_importance.png',
            'artifacts/classification_report.json']


def train_model(data_dir, dagshub_owner=None, dagshub_repo=None):
    """Main training function"""
    
    print("="*60)
    print("MLPROJECT TRAINING PIPELINE")
    print("="*60)
    
    # Initialize DagsHub if credentials provided
    if dagshub_owner and dagshub_repo:
        print(f"\nInitializing DagsHub: {dagshub_owner}/{dagshub_repo}")
        try:
            dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)
        except Exception as e:
            print(f"Warning: Could not initialize DagsHub: {e}")
    
    # Set experiment
    mlflow.set_experiment("Heart_Disease_MLProject")
    
    with mlflow.start_run(run_name="MLProject_HeartDisease_CI"):
        
        # Load data
        X_train, X_test, y_train, y_test = load_preprocessed_data(data_dir)
        
        # Hyperparameter tuning
        best_model, best_params = perform_hyperparameter_tuning(X_train, y_train)
        
        # Log parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("data_dir", data_dir)
        mlflow.log_param("dataset", "Heart Disease (Cleveland)")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Calculate and log metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        mlflow.log_metric("cv_score_mean", cv_scores.mean())
        mlflow.log_metric("cv_score_std", cv_scores.std())
        
        # Create and log artifacts
        artifact_paths = create_artifacts(y_test, y_pred, best_model, X_train.columns.tolist())
        for path in artifact_paths:
            mlflow.log_artifact(path)
        
        # Log model
        mlflow.sklearn.log_model(
            best_model,
            "model",
            registered_model_name="HeartDiseaseClassifier_MLProject"
        )
        
        # Save model locally
        os.makedirs('model_output', exist_ok=True)
        joblib.dump(best_model, 'model_output/model.pkl')
        mlflow.log_artifact('model_output/model.pkl')
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Dataset: Heart Disease (Cleveland)")
        print(f"Model: RandomForestClassifier")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print("="*60)


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/preprocessed"
    
    dagshub_owner = sys.argv[2] if len(sys.argv) > 2 else None
    dagshub_repo = sys.argv[3] if len(sys.argv) > 3 else None
    
    train_model(data_dir, dagshub_owner, dagshub_repo)
