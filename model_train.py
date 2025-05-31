"""
Credit Card Fraud Detection Model Training Script
This script trains a logistic regression model for fraud detection
and saves it as 'logistic_regression_model.pkl' for use with your FastAPI server
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def load_and_explore_data(file_path):
    """Load and explore the credit card fraud dataset"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFeatures: {df.columns.tolist()}")
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum().sum()}")
    
    # Class distribution
    fraud_count = df['Class'].sum()
    normal_count = len(df) - fraud_count
    print(f"\nClass distribution:")
    print(f"Normal transactions: {normal_count} ({normal_count/len(df)*100:.2f}%)")
    print(f"Fraudulent transactions: {fraud_count} ({fraud_count/len(df)*100:.2f}%)")
    
    return df

def preprocess_data(df):
    """Preprocess the data for model training"""
    print("\nPreprocessing data...")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale Time and Amount features (V1-V28 are already scaled)
    scaler = StandardScaler()
    X['scaled_amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
    X['scaled_time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))
    
    # Drop original Time and Amount columns
    X = X.drop(['Time', 'Amount'], axis=1)
    
    # Reorder columns to match server expectations
    feature_cols = ['scaled_amount', 'scaled_time'] + [f'V{i}' for i in range(1, 29)]
    X = X[feature_cols]
    
    print(f"Final feature columns: {X.columns.tolist()}")
    
    return X, y, scaler

def save_processed_data(X_train, y_train, X_test, y_test, 
                       train_filename='processed_training_data.csv',
                       test_filename='processed_test_data.csv'):
    """Save the processed training and test data to CSV files"""
    print(f"\nSaving processed training data to {train_filename}...")
    
    # Combine features and target for training data
    train_data = X_train.copy()
    train_data['Class'] = y_train
    train_data.to_csv(train_filename, index=False)
    
    print(f"Saving processed test data to {test_filename}...")
    
    # Combine features and target for test data
    test_data = X_test.copy()
    test_data['Class'] = y_test
    test_data.to_csv(test_filename, index=False)
    
    print(f"✅ Processed data saved successfully!")
    print(f"   Training data shape: {train_data.shape}")
    print(f"   Test data shape: {test_data.shape}")
    print(f"   Training data saved to: {train_filename}")
    print(f"   Test data saved to: {test_filename}")

def save_resampled_data(X_resampled, y_resampled, filename='resampled_training_data.csv'):
    """Save the resampled training data to CSV file"""
    print(f"\nSaving resampled training data to {filename}...")
    
    # Combine features and target for resampled data
    resampled_data = X_resampled.copy()
    resampled_data['Class'] = y_resampled
    resampled_data.to_csv(filename, index=False)
    
    print(f"✅ Resampled data saved successfully!")
    print(f"   Resampled data shape: {resampled_data.shape}")
    print(f"   Normal transactions: {sum(y_resampled==0)}")
    print(f"   Fraudulent transactions: {sum(y_resampled==1)}")
    print(f"   Resampled data saved to: {filename}")

def handle_imbalanced_data(X_train, y_train, method='undersample'):
    """Handle imbalanced dataset using various techniques"""
    print(f"\nHandling imbalanced data using {method}...")
    
    if method == 'undersample':
        # Random undersampling
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    elif method == 'oversample':
        # SMOTE oversampling
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    elif method == 'combined':
        # Combination of over and under sampling
        over = SMOTE(sampling_strategy=0.5, random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
        pipeline = Pipeline([('o', over), ('u', under)])
        X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
    else:
        # No resampling
        X_resampled, y_resampled = X_train, y_train
    
    print(f"Resampled training set - Normal: {sum(y_resampled==0)}, Fraud: {sum(y_resampled==1)}")
    
    return X_resampled, y_resampled

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and compare performance"""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'Logistic Regression (L1)': LogisticRegression(
            penalty='l1',
            solver='liblinear',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"\nClassification Report:\n{results[name]['classification_report']}")
        
    return results

def plot_results(results, X_test, y_test):
    """Plot model performance metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: ROC Curves
    ax1 = axes[0, 0]
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        ax1.plot(fpr, tpr, label=f"{name} (AUC: {result['roc_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confusion Matrix for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    ax2 = axes[0, 1]
    cm = results[best_model_name]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title(f'Confusion Matrix - {best_model_name}')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # Plot 3: Feature Importance (if Random Forest)
    ax3 = axes[1, 0]
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        ax3.barh(feature_importance['feature'], feature_importance['importance'])
        ax3.set_xlabel('Importance')
        ax3.set_title('Top 15 Feature Importances (Random Forest)')
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, 'Feature importance only available for Random Forest', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Model Comparison
    ax4 = axes[1, 1]
    model_names = list(results.keys())
    roc_scores = [results[name]['roc_auc'] for name in model_names]
    ax4.bar(model_names, roc_scores)
    ax4.set_ylabel('ROC AUC Score')
    ax4.set_title('Model Performance Comparison')
    ax4.set_ylim(0.9, 1.0)
    for i, score in enumerate(roc_scores):
        ax4.text(i, score + 0.005, f'{score:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('fraud_detection_model_performance.png', dpi=300)
    plt.show()

def save_model_and_scaler(model, scaler, model_filename='logistic_regression_model.pkl', 
                         scaler_filename='scaler.pkl'):
    """Save the trained model and scaler"""
    print(f"\nSaving model to {model_filename}...")
    joblib.dump(model, model_filename)
    
    print(f"Saving scaler to {scaler_filename}...")
    joblib.dump(scaler, scaler_filename)
    
    print("Model and scaler saved successfully!")

def main():
    """Main training pipeline"""
    # Configuration
    CSV_FILE = 'creditcard.csv'  # Update with your file path
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    RESAMPLING_METHOD = 'none'  # Options: 'none', 'undersample', 'oversample', 'combined'
    SAVE_PROCESSED_DATA = True  # Set to True to save processed data
    
    # Load and explore data
    df = load_and_explore_data(CSV_FILE)
    
    # Preprocess data
    X, y, scaler = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Save processed data if requested
    if SAVE_PROCESSED_DATA:
        save_processed_data(X_train, y_train, X_test, y_test)
    
    # Handle imbalanced data (optional)
    X_train_resampled, y_train_resampled = handle_imbalanced_data(
        X_train, y_train, method=RESAMPLING_METHOD
    )
    
    # Save resampled data if resampling was applied
    if RESAMPLING_METHOD != 'none' and SAVE_PROCESSED_DATA:
        save_resampled_data(X_train_resampled, y_train_resampled)
    
    # Train models
    results = train_models(X_train_resampled, y_train_resampled, X_test, y_test)
    
    # Plot results
    plot_results(results, X_test, y_test)
    
    # Select best model (Logistic Regression for compatibility with your server)
    # Using the basic Logistic Regression as it's what your server expects
    best_model = results['Logistic Regression']['model']
    
    # Perform cross-validation on the best model
    print("\nPerforming 5-fold cross-validation on Logistic Regression...")
    cv_scores = cross_val_score(best_model, X, y, cv=StratifiedKFold(5), scoring='roc_auc')
    print(f"Cross-validation ROC AUC scores: {cv_scores}")
    print(f"Mean CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save the model and scaler
    save_model_and_scaler(best_model, scaler)
    
    # Test loading the model
    print("\nTesting model loading...")
    loaded_model = joblib.load('logistic_regression_model.pkl')
    test_prediction = loaded_model.predict(X_test[:5])
    print(f"Test predictions on first 5 samples: {test_prediction}")
    
    print("\n✅ Training completed successfully!")
    print(f"Model saved as 'logistic_regression_model.pkl'")
    print(f"You can now run your FastAPI server and it will use this trained model.")
    
    if SAVE_PROCESSED_DATA:
        print("\n📁 Processed data files created:")
        print("   - processed_training_data.csv")
        print("   - processed_test_data.csv")
        if RESAMPLING_METHOD != 'none':
            print("   - resampled_training_data.csv")

if __name__ == "__main__":
    # If you have the sample data I provided, save it as a CSV first
    
    # with open('creditcard.csv', 'w') as f:
    #     f.write(sample_data)
    
    main()
    
    print("Please update the CSV_FILE path in main() and run this script with your full dataset.")
    print("\nRequired packages:")
    print("pip install pandas scikit-learn joblib matplotlib seaborn imbalanced-learn")