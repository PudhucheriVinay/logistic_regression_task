import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve

def plot_sigmoid():
    x = np.linspace(-10, 10, 100)
    sigmoid = 1 / (1 + np.exp(-x))
    plt.figure(figsize=(6,4))
    plt.plot(x, sigmoid, label='Sigmoid function')
    plt.title('Sigmoid Function')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid(True)
    plt.legend()
    plt.savefig('sigmoid_function.png')
    plt.show()

def main():
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:,1]

    # Default threshold 0.5
    y_pred = (y_probs >= 0.5).astype(int)

    # Evaluation metrics
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)

    print("Confusion Matrix:")
    print(cm)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.show()

    # Threshold tuning example
    thresholds_to_try = [0.3, 0.5, 0.7]
    for thresh in thresholds_to_try:
        y_pred_thresh = (y_probs >= thresh).astype(int)
        precision_t = precision_score(y_test, y_pred_thresh)
        recall_t = recall_score(y_test, y_pred_thresh)
        print(f"Threshold: {thresh} - Precision: {precision_t:.3f}, Recall: {recall_t:.3f}")

    # Plot sigmoid function
    plot_sigmoid()

if __name__ == "__main__":
    main()
