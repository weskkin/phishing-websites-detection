import pandas as pd
from sklearn.linear_model import LogisticRegression
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

# Load preprocessed data
X_train = pd.read_csv('../data/processed/X_train_scaled.csv')
X_test = pd.read_csv('../data/processed/X_test_scaled.csv')
y_train = pd.read_csv('../data/processed/y_train.csv')['label']
y_test = pd.read_csv('../data/processed/y_test.csv')['label']
print("Loaded preprocessed data")

# Initialize and train Logistic Regression
lr_model = LogisticRegression(random_state=43, max_iter=1000) # Increased max iterations for convergence
print("Created logistic regression model")
lr_model.fit(X_train, y_train)
print("Trained the logistic regression model")

# Prediction
print("Predicting from test data")
y_pred = lr_model.predict(X_test) # Now that you’ve learned from the training data, what do you think the answers are for the test data, The result is stored in y_pred, which is a list of 0s and 1s (predicted labels)

# Probabilities for ROC-AUC
y_proba = lr_model.predict_proba(X_test)[:, 1] # probability of being legitimate

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred) # How many predictions were correct (overall %)
precision = precision_score(y_test, y_pred) # How many predicted “1” were actually “1”
recall = recall_score(y_test, y_pred) # How many actual “1” were found correctly
f1 = f1_score(y_test, y_pred) # Balance between precision and recall
roc_auc = roc_auc_score(y_test, y_proba) # How well the model distinguishes between classes (based on probability scores)

print("Logistic Regression Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba) # Calculates False Positive Rate and True Positive Rate at different thresholds used for drawing a ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--') # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('../reports/roc_curve_lr.png')  # Create a 'reports' folder for plots
plt.close()

# Save the model
joblib.dump(lr_model, '../models/logistic_regression.pkl')