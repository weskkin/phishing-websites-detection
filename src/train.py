import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import joblib
import json



def train_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Trains a model

        Evaluates it (accuracy, precision, etc.)

        Saves the model as .pkl file

        Returns the metrics in a dictionary"""

    print(f"{model_name} model initialized")

    # Train
    print(f"training {model_name}")
    model.fit(X_train, y_train)

    # Predictions
    print("Predicting from test data")
    y_pred = model.predict(X_test)
    # Probabilities for ROC-AUC
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)

    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred), # How many predictions were correct (overall %)
        "Precision": precision_score(y_test, y_pred), # How many predicted “1” were actually “1”
        "Recall": recall_score(y_test, y_pred), # How many actual “1” were found correctly
        "F1": f1_score(y_test, y_pred),  # Balance between precision and recall
        "ROC-AUC": roc_auc_score(y_test, y_proba) if hasattr(model, "predict_proba") else None # How well the model distinguishes between classes (based on probability scores)
    }
    print(f"{model_name} Metrics:")
    print(f"Accuracy: {metrics["Accuracy"]:.4f}")
    print(f"Precision: {metrics["Precision"]:.4f}")
    print(f"Recall: {metrics["Recall"]:.4f}")
    print(f"F1-Score: {metrics["F1"]:.4f}")
    print(f"ROC-AUC: {metrics["ROC-AUC"]:.4f}")


    # Plot ROC Curve
    if hasattr(model, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_test, y_proba) # Calculates False Positive Rate and True Positive Rate at different thresholds used for drawing a ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics['ROC-AUC']:.2f})')
        plt.plot([0, 1], [0, 1], 'k--') # Diagonal reference line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(f'../reports/roc_curve_{model_name}.png')  # Create a 'reports' folder for plots
        plt.close()
        print(f"Roc curve plot generated and saved at reports/roc_curve_{model_name}.png")

    # Save model
    joblib.dump(model, f'../models/{model_name}.pkl')
    print(f"{model_name} model saved at models/{model_name}.pkl") # change later
    return metrics

# Load preprocessed data
X_train = pd.read_csv('../data/processed/X_train_scaled.csv')
X_test = pd.read_csv('../data/processed/X_test_scaled.csv')
y_train = pd.read_csv('../data/processed/y_train.csv')['label']
y_test = pd.read_csv('../data/processed/y_test.csv')['label']
print("Loaded preprocessed data")

print("Initializing models")
# Initialize models
models = {
    "LogisticRegression": LogisticRegression(random_state=43, max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100)
}

# Train all models and save results
results = {}
for name, model in models.items():
    results[name] = train_evaluate_model(model, name, X_train, X_test, y_train, y_test)

# Save results for GUI
with open('../reports/metrics.json', 'w') as f:
    json.dump(results, f, indent=4)

print("All metrics results saved at reports/metrics.json")

print("All models trained and saved!")