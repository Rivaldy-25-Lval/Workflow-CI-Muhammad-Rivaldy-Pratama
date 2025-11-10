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
    
    X_train = train_data.drop('quality_category', axis=1)
    y_train = train_data['quality_category']
    X_test = test_data.drop('quality_category', axis=1)
    y_test = test_data['quality_category']
    
    print(f"✅ Data loaded! Train: {len(X_train)}, Test: {len(X_test)}")
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
    """Calculate all metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'log_loss': log_loss(y_true, y_pred_proba),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
    }
    
    try:
        metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, 
                                               multi_class='ovr', average='weighted')
    except:
        metrics['roc_auc_ovr'] = 0.0
    
    return metrics


def create_artifacts(y_true, y_pred, model, feature_names):
    """Create and save artifacts"""
    os.makedirs('artifacts', exist_ok=True)
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Medium', 'High'],
                yticklabels=['Low', 'Medium', 'High'])
    plt.title('Confusion Matrix', fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('artifacts/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance
    plt.figure(figsize=(12, 8))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[indices], color='skyblue')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], 
               rotation=45, ha='right')
    plt.title('Feature Importances', fontweight='bold')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('artifacts/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return ['artifacts/confusion_matrix.png', 'artifacts/feature_importance.png']


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
    mlflow.set_experiment("Wine_Quality_MLProject")
    
    with mlflow.start_run(run_name="MLProject_CI_Training"):
        
        # Load data
        X_train, X_test, y_train, y_test = load_preprocessed_data(data_dir)
        
        # Hyperparameter tuning
        best_model, best_params = perform_hyperparameter_tuning(X_train, y_train)
        
        # Log parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("data_dir", data_dir)
        
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
            registered_model_name="WineQualityClassifier_MLProject"
        )
        
        # Save model locally
        os.makedirs('model_output', exist_ok=True)
        joblib.dump(best_model, 'model_output/model.pkl')
        mlflow.log_artifact('model_output/model.pkl')
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_weighted']:.4f}")
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
