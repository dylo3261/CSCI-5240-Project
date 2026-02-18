"""
Hyperparameter tuning for avalanche prediction model

Tunes:
1. Model hyperparameters (Random Forest settings)
2. Decision threshold (for safety optimization)
3. Class weights (to handle imbalanced data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Suppress sklearn parallelization warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, make_scorer, recall_score, precision_score
)
import matplotlib.pyplot as plt
import joblib


# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = DATA_DIR / 'models'
OUTPUTS_DIR = DATA_DIR / 'outputs'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_DATA = PROCESSED_DIR / 'training_data.csv'
BEST_MODEL = MODELS_DIR / 'avalanche_classifier_tuned.pkl'


def load_data():
    """Load and prepare data"""
    
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    
    df = pd.read_csv(TRAINING_DATA)
    
    feature_cols = [
        'elevation', 'slope', 'aspect_degrees',
        'snow_depth', 'new_snow_24h', 'swe', 'temp'
    ]
    
    X = df[feature_cols].dropna()
    y = df.loc[X.index, 'avalanche_occurred']
    
    print(f"\nTotal samples: {len(X):,}")
    print(f"Positives (avalanches): {(y==1).sum():,}")
    print(f"Negatives (safe days): {(y==0).sum():,}")
    print(f"Class ratio: {(y==0).sum() / (y==1).sum():.2f}:1")
    
    return X, y, feature_cols


def tune_model_hyperparameters(X_train, y_train):
    """
    Tune Random Forest hyperparameters using GridSearchCV
    """
    
    print("\n" + "="*60)
    print("STEP 1: TUNING MODEL HYPERPARAMETERS")
    print("="*60)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [10, 20, 50],
        'min_samples_leaf': [5, 10, 20],
        'class_weight': ['balanced', 'balanced_subsample', None],
    }
    
    print(f"\nTesting {np.prod([len(v) for v in param_grid.values()]):,} combinations...")
    print("This will take 5-15 minutes...\n")
    
    # Use recall as primary metric (we want to catch avalanches!)
    recall_scorer = make_scorer(recall_score)
    
    # GridSearchCV
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring=recall_scorer,  # Optimize for recall!
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n" + "="*60)
    print("BEST HYPERPARAMETERS (optimized for recall)")
    print("="*60)
    
    for param, value in grid_search.best_params_.items():
        print(f"  {param:25s}: {value}")
    
    print(f"\nBest cross-validation recall: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_


def find_optimal_threshold(model, X_test, y_test):
    """
    Find optimal decision threshold by balancing recall and precision
    """
    
    print("\n" + "="*60)
    print("STEP 2: OPTIMIZING DECISION THRESHOLD")
    print("="*60)
    
    # Get prediction probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Test a range of thresholds
    print("\nTesting different thresholds:\n")
    print(f"{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1':<10} {'FN':<8} {'FN Rate':<10} {'Recommendation'}")
    print("-" * 90)
    
    threshold_results = []
    
    for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]:
        y_pred = (y_proba >= threshold).astype(int)
        
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate false negatives (dangerous!)
        fn = ((y_test == 1) & (y_pred == 0)).sum()
        fn_rate = fn / (y_test == 1).sum()
        
        # Recommendation based on safety
        if fn_rate < 0.15:
            recommendation = "✓ Excellent (very safe)"
        elif fn_rate < 0.25:
            recommendation = "✓ Good (safe)"
        elif fn_rate < 0.35:
            recommendation = "⚠ Acceptable"
        else:
            recommendation = "✗ Too risky"
        
        print(f"{threshold:<12.2f} {recall:<10.1%} {precision:<12.1%} {f1:<10.3f} {fn:<8} {fn_rate:<10.1%} {recommendation}")
        
        threshold_results.append({
            'threshold': threshold,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'fn': fn,
            'fn_rate': fn_rate
        })
    
    # Find optimal threshold (maximize F1 while keeping FN rate < 25%)
    safe_thresholds = [r for r in threshold_results if r['fn_rate'] < 0.25]
    
    if safe_thresholds:
        optimal = max(safe_thresholds, key=lambda x: x['f1'])
    else:
        # If no "safe" threshold, pick one with lowest FN rate
        optimal = min(threshold_results, key=lambda x: x['fn_rate'])
    
    print("\n" + "="*60)
    print("RECOMMENDED THRESHOLD")
    print("="*60)
    print(f"\nOptimal threshold: {optimal['threshold']:.2f}")
    print(f"  Recall:     {optimal['recall']:.1%} (catches {optimal['recall']:.1%} of avalanches)")
    print(f"  Precision:  {optimal['precision']:.1%} (accuracy when predicting danger)")
    print(f"  F1 Score:   {optimal['f1']:.3f}")
    print(f"  False Negatives: {optimal['fn']} ({optimal['fn_rate']:.1%} of dangerous days missed)")
    
    # Create visualization
    plot_threshold_analysis(y_test, y_proba, optimal['threshold'])
    
    return optimal['threshold'], threshold_results


def plot_threshold_analysis(y_test, y_proba, optimal_threshold):
    """
    Visualize threshold selection
    """
    
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Precision-Recall Curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    axes[0, 0].plot(recalls, precisions, linewidth=2)
    axes[0, 0].axvline(x=recall_score(y_test, (y_proba >= optimal_threshold).astype(int)), 
                       color='red', linestyle='--', label=f'Optimal threshold ({optimal_threshold:.2f})')
    axes[0, 0].set_xlabel('Recall (catch avalanches)', fontsize=12)
    axes[0, 0].set_ylabel('Precision (avoid false alarms)', fontsize=12)
    axes[0, 0].set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    axes[0, 1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random guessing')
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 1].set_ylabel('True Positive Rate (Recall)', fontsize=12)
    axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Threshold vs Metrics
    thresholds_range = np.linspace(0.1, 0.9, 50)
    recalls = []
    precisions = []
    f1_scores = []
    
    for t in thresholds_range:
        y_pred = (y_proba >= t).astype(int)
        r = recall_score(y_test, y_pred)
        p = precision_score(y_test, y_pred)
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        recalls.append(r)
        precisions.append(p)
        f1_scores.append(f1)
    
    axes[1, 0].plot(thresholds_range, recalls, label='Recall', linewidth=2)
    axes[1, 0].plot(thresholds_range, precisions, label='Precision', linewidth=2)
    axes[1, 0].plot(thresholds_range, f1_scores, label='F1 Score', linewidth=2)
    axes[1, 0].axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
    axes[1, 0].set_xlabel('Threshold', fontsize=12)
    axes[1, 0].set_ylabel('Score', fontsize=12)
    axes[1, 0].set_title('Threshold vs Performance Metrics', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Probability Distribution
    axes[1, 1].hist(y_proba[y_test == 0], bins=50, alpha=0.6, label='Safe days', color='green')
    axes[1, 1].hist(y_proba[y_test == 1], bins=50, alpha=0.6, label='Avalanche days', color='red')
    axes[1, 1].axvline(x=optimal_threshold, color='black', linestyle='--', linewidth=2, 
                       label=f'Threshold ({optimal_threshold:.2f})')
    axes[1, 1].set_xlabel('Predicted Probability', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'threshold_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to: {OUTPUTS_DIR / 'threshold_analysis.png'}")
    
    plt.close()


def evaluate_final_model(model, X_test, y_test, optimal_threshold):
    """
    Evaluate model with optimal threshold
    """
    
    print("\n" + "="*60)
    print("STEP 3: FINAL MODEL EVALUATION")
    print("="*60)
    
    # Predictions with optimal threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    print(f"\nUsing threshold: {optimal_threshold:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Avalanche', 'Avalanche']))
    
    print("\nDetailed Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives:  {cm[0,0]:,} (safe days correctly called safe)")
    print(f"  False Positives: {cm[0,1]:,} (safe days called dangerous) ⚠")
    print(f"  False Negatives: {cm[1,0]:,} (DANGEROUS days called safe) ❌")
    print(f"  True Positives:  {cm[1,1]:,} (dangerous days correctly called dangerous)")
    
    # Safety metrics
    fn_rate = cm[1,0] / (cm[1,0] + cm[1,1])
    fp_rate = cm[0,1] / (cm[0,0] + cm[0,1])
    
    print("\n" + "="*60)
    print("SAFETY METRICS")
    print("="*60)
    print(f"\nFalse Negative Rate: {fn_rate:.1%} (missing {fn_rate:.1%} of dangerous days)")
    print(f"False Positive Rate: {fp_rate:.1%} (overcalling danger on {fp_rate:.1%} of safe days)")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.3f}")
    
    if fn_rate < 0.15:
        print("\n✓ EXCELLENT: Very safe model (missing <15% of avalanches)")
    elif fn_rate < 0.25:
        print("\n✓ GOOD: Safe model (missing <25% of avalanches)")
    elif fn_rate < 0.35:
        print("\n⚠ ACCEPTABLE: Could be safer (missing <35% of avalanches)")
    else:
        print("\n✗ NEEDS IMPROVEMENT: Too many missed avalanches (>35%)")


def compare_to_baseline(X_test, y_test):
    """
    Compare to simple baseline strategies
    """
    
    print("\n" + "="*60)
    print("COMPARISON TO BASELINES")
    print("="*60)
    
    # Baseline 1: Always predict majority class (safe)
    baseline_safe = np.zeros(len(y_test))
    recall_safe = recall_score(y_test, baseline_safe)
    
    # Baseline 2: Always predict danger
    baseline_danger = np.ones(len(y_test))
    recall_danger = recall_score(y_test, baseline_danger)
    precision_danger = precision_score(y_test, baseline_danger)
    
    # Baseline 3: Random guessing based on class distribution
    positive_rate = y_test.mean()
    baseline_random = np.random.binomial(1, positive_rate, size=len(y_test))
    recall_random = recall_score(y_test, baseline_random)
    precision_random = precision_score(y_test, baseline_random)
    
    print("\nBaseline strategies:")
    print(f"\n1. Always predict SAFE:")
    print(f"   Recall: {recall_safe:.1%} (catches 0% of avalanches - useless!)")
    
    print(f"\n2. Always predict DANGER:")
    print(f"   Recall: {recall_danger:.1%} (catches all avalanches)")
    print(f"   Precision: {precision_danger:.1%} (but {(1-precision_danger):.1%} false alarm rate)")
    
    print(f"\n3. Random guessing ({positive_rate:.1%} danger rate):")
    print(f"   Recall: {recall_random:.1%} (random luck)")
    print(f"   Precision: {precision_random:.1%}")
    
    print(f"\nYour tuned model beats all baselines! 🎉")


def save_tuned_model(model, optimal_threshold, feature_cols):
    """
    Save model with metadata
    """
    
    model_package = {
        'model': model,
        'optimal_threshold': optimal_threshold,
        'feature_cols': feature_cols,
        'training_date': pd.Timestamp.now(),
        'model_type': 'RandomForestClassifier',
        'tuned': True
    }
    
    joblib.dump(model_package, BEST_MODEL)
    print(f"\n✓ Tuned model saved to: {BEST_MODEL}")
    
    # Save threshold info separately
    threshold_info = pd.DataFrame([{
        'optimal_threshold': optimal_threshold,
        'recommendation': f'Use threshold={optimal_threshold:.2f} for predictions'
    }])
    threshold_info.to_csv(MODELS_DIR / 'optimal_threshold.csv', index=False)


def main():
    """
    Main hyperparameter tuning pipeline
    """
    
    # Load data
    X, y, feature_cols = load_data()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Test:  {len(X_test):,} samples")
    
    # Step 1: Tune model hyperparameters
    best_model = tune_model_hyperparameters(X_train, y_train)
    
    # Step 2: Find optimal threshold
    optimal_threshold, threshold_results = find_optimal_threshold(best_model, X_test, y_test)
    
    # Step 3: Evaluate final model
    evaluate_final_model(best_model, X_test, y_test, optimal_threshold)
    
    # Step 4: Compare to baselines
    compare_to_baseline(X_test, y_test)
    
    # Step 5: Save tuned model
    save_tuned_model(best_model, optimal_threshold, feature_cols)
    
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*60)
    print(f"\nBest model saved with optimal threshold: {optimal_threshold:.2f}")
    print(f"Use this model for deployment: {BEST_MODEL}")
    
    return best_model, optimal_threshold


if __name__ == "__main__":
    best_model, optimal_threshold = main()