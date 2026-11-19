import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

print("=" * 70)
print(" PRACTICAL EXAM: TWO-CLASS CLASSIFICATION USING NAIVE BAYES AND SVM")
print("=" * 70)

# 1. Ingestion
local_path = os.path.join(os.path.dirname(__file__), "datasets", "pima_diabetes.csv")
if os.path.exists(local_path):
    df = pd.read_csv(local_path)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    dataset_name = "Pima Indians Diabetes (0: Healthy, 1: Diabetic)"
    target_names = ["Healthy (0)", "Diabetic (1)"]
else:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    dataset_name = "Breast Cancer Diagnostic (0: Malignant, 1: Benign)"
    target_names = ["Malignant (0)", "Benign (1)"]

print(f"\n[1] Dataset Loaded: {dataset_name}")
print(f"    Dimensions   : {X.shape[0]} observations, {X.shape[1]} predictor features")
print(f"    Classes      : {target_names[0]} vs {target_names[1]}")

# 2. Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)
print(f"\n[2] Partition: 80% Train ({len(X_train)} samples), 20% Test ({len(X_test)} samples)")

# 3. Model 1: Gaussian Naive Bayes
nb = GaussianNB().fit(X_tr_s, y_train)
pred_nb = nb.predict(X_te_s)
prob_nb = nb.predict_proba(X_te_s)[:, 1]

# 4. Model 2: Support Vector Machine (SVM)
svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42).fit(X_tr_s, y_train)
pred_svm = svm.predict(X_te_s)
prob_svm = svm.predict_proba(X_te_s)[:, 1]

# 5. Side-by-Side Performance Comparison
table = pd.DataFrame({
    "Algorithm": ["Gaussian Naive Bayes", "Support Vector Machine (SVM)"],
    "Accuracy (%)": [round(accuracy_score(y_test, pred_nb)*100, 2), round(accuracy_score(y_test, pred_svm)*100, 2)],
    "Precision": [round(precision_score(y_test, pred_nb), 4), round(precision_score(y_test, pred_svm), 4)],
    "Recall": [round(recall_score(y_test, pred_nb), 4), round(recall_score(y_test, pred_svm), 4)],
    "F1-Score": [round(f1_score(y_test, pred_nb), 4), round(f1_score(y_test, pred_svm), 4)],
    "ROC-AUC": [round(roc_auc_score(y_test, prob_nb), 4), round(roc_auc_score(y_test, prob_svm), 4)]
})

print("\n" + "=" * 70)
print(" MODEL PERFORMANCE COMPARISON TABLE")
print("=" * 70)
print(table.to_string(index=False))

print("\n" + "=" * 70)
print(" CONFUSION MATRICES (TN, FP / FN, TP)")
print("=" * 70)
print("Naive Bayes Confusion Matrix:\n", confusion_matrix(y_test, pred_nb))
print("\nSVM (RBF Kernel) Confusion Matrix:\n", confusion_matrix(y_test, pred_svm))

print("\n" + "=" * 70)
print(" LIVE PREDICTION ON UNSEEN TEST QUERY")
print("=" * 70)
sample = X_te_s[0:1]
print(f"True Class        : [{target_names[int(y_test.iloc[0])]}]")
print(f"Naive Bayes Pred  : [{target_names[int(nb.predict(sample)[0])]}] (Confidence: {max(nb.predict_proba(sample)[0])*100:.1f}%)")
print(f"SVM Prediction    : [{target_names[int(svm.predict(sample)[0])]}] (Confidence: {max(svm.predict_proba(sample)[0])*100:.1f}%)")
print("=" * 70)
print("Demonstration Completed Successfully!")
