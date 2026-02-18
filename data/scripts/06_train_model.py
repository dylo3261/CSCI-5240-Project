"""
Train avalanche prediction model with feature importance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, r2_score
import joblib

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = DATA_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_DATA = PROCESSED_DIR / 'training_data.csv'
CLASSIFIER_MODEL = MODELS_DIR / 'avalanche_classifier.pkl'
REGRESSOR_MODEL = MODELS_DIR / 'avalanche_size_regressor.pkl'


def load_and_prepare_data():
    """Load training data and prepare features"""
    
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Load data
    df = pd.read_csv(TRAINING_DATA)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"\nTotal samples: {len(df):,}")
    print(f"\nClass distribution:")
    print(df['avalanche_occurred'].value_counts())
    
    # Define feature columns
    feature_cols = [
        'elevation',
        'slope',
        'aspect_degrees',
        'snow_depth',
        'new_snow_24h',
        'swe',
        'temp'
    ]
    
    # Check for missing values
    print(f"\nMissing values:")
    for col in feature_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
    # Drop rows with missing features
    df_clean = df[feature_cols + ['avalanche_occurred', 'avalanche_size']].dropna()
    print(f"\nAfter removing missing values: {len(df_clean):,} samples")
    
    # Separate features and targets
    X = df_clean[feature_cols]
    y_class = df_clean['avalanche_occurred']
    y_size = df_clean['avalanche_size']
    
    return X, y_class, y_size, feature_cols


def train_classifier(X, y, feature_cols):
    """Train binary classifier (avalanche yes/no)"""
    
    print("\n" + "="*60)
    print("TRAINING CLASSIFIER (AVALANCHE YES/NO)")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    
    # Train Random Forest (good for SHAP)
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*60)
    print("CLASSIFIER EVALUATION")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Avalanche', 'Avalanche']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives:  {cm[0,0]:,}")
    print(f"False Positives: {cm[0,1]:,}")
    print(f"False Negatives: {cm[1,0]:,}")
    print(f"True Positives:  {cm[1,1]:,}")
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba):.3f}")
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Built-in)")
    print("="*60)
    
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n", importances.to_string(index=False))
    
    # Save model
    joblib.dump(model, CLASSIFIER_MODEL)
    print(f"\n✓ Model saved to: {CLASSIFIER_MODEL}")
    
    return model, X_test, y_test


def train_regressor(X, y, feature_cols):
    """Train size regressor (D1-D5 scale)"""
    
    print("\n" + "="*60)
    print("TRAINING REGRESSOR (AVALANCHE SIZE)")
    print("="*60)
    
    # Only train on positive examples (where avalanches occurred)
    mask = y > 0
    X_pos = X[mask]
    y_pos = y[mask]
    
    print(f"\nPositive samples for size prediction: {len(X_pos):,}")
    
    if len(X_pos) < 100:
        print("Not enough positive samples for size prediction")
        return None, None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_pos, y_pos, test_size=0.2, random_state=42
    )
    
    # Train Gradient Boosting
    print("\nTraining Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    print("\n" + "="*60)
    print("REGRESSOR EVALUATION")
    print("="*60)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nRMSE: {rmse:.3f} (error in size classes)")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")
    
    # Feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Size Prediction)")
    print("="*60)
    
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n", importances.to_string(index=False))
    
    # Save model
    joblib.dump(model, REGRESSOR_MODEL)
    print(f"\n✓ Model saved to: {REGRESSOR_MODEL}")
    
    return model, X_test, y_test


def main():
    """Main training pipeline"""
    
    # Load data
    X, y_class, y_size, feature_cols = load_and_prepare_data()
    
    # Train classifier
    classifier, X_test_class, y_test_class = train_classifier(X, y_class, feature_cols)
    
    # Train regressor
    regressor, X_test_reg, y_test_reg = train_regressor(X, y_size, feature_cols)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModels saved in: {MODELS_DIR}")
    
    return classifier, regressor


if __name__ == "__main__":
    classifier, regressor = main()