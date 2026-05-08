# ============================================================
# PROJECT 4 — Loan Approval Prediction System
# VTU 2022 Scheme | Data Science Lab | Dept. of ECE
# ============================================================

# STEP 1 — Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

# STEP 2 — Create Sample Dataset
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'Gender':            np.random.choice(['Male', 'Female'], n),
    'Married':           np.random.choice(['Yes', 'No'], n),
    'Dependents':        np.random.choice(['0', '1', '2', '3+'], n),
    'Education':         np.random.choice(['Graduate', 'Not Graduate'], n),
    'ApplicantIncome':   np.random.randint(2000, 15000, n),
    'CoapplicantIncome': np.random.randint(0, 5000, n),
    'LoanAmount':        np.random.randint(50, 500, n),
    'Credit_History':    np.random.choice([0, 1], n, p=[0.2, 0.8]),
    'Property_Area':     np.random.choice(['Urban', 'Rural', 'Semiurban'], n),
})

data['Loan_Status'] = (
    (data['Credit_History'] * 0.5 +
     (data['ApplicantIncome'] > 5000).astype(int) * 0.3 +
     (data['Education'] == 'Graduate').astype(int) * 0.2 +
     np.random.uniform(0, 0.3, n)) > 0.45
).astype(int)

print("Dataset Shape:", data.shape)
print(data['Loan_Status'].value_counts())

# STEP 3 — EDA
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

sns.countplot(data=data, x='Credit_History', hue='Loan_Status',
              palette={0: 'salmon', 1: 'steelblue'}, ax=axes[0])
axes[0].set_title('Credit History vs Loan Status')
axes[0].legend(['Rejected', 'Approved'])

sns.boxplot(data=data, x='Loan_Status', y='ApplicantIncome',
            palette={0: 'salmon', 1: 'steelblue'}, ax=axes[1])
axes[1].set_title('Income vs Loan Status')
axes[1].set_xticklabels(['Rejected', 'Approved'])

sns.countplot(data=data, x='Education', hue='Loan_Status',
              palette={0: 'salmon', 1: 'steelblue'}, ax=axes[2])
axes[2].set_title('Education vs Loan Status')
axes[2].legend(['Rejected', 'Approved'])

plt.tight_layout()
plt.show()

# STEP 4 — Preprocessing
le = LabelEncoder()
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Property_Area']
for col in cat_cols:
    data[col] = le.fit_transform(data[col])

X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# STEP 5 — Train and Compare Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Naive Bayes':         GaussianNB(),
    'SVM':                 SVC(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:25s} → Accuracy: {acc*100:.2f}%")

# STEP 6 — Accuracy Chart
plt.figure(figsize=(10, 5))
bars = plt.bar(results.keys(), [v * 100 for v in results.values()],
               color=['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f'])
plt.ylim(60, 100)
plt.title('Model Accuracy Comparison — Loan Approval Prediction', fontsize=13)
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=15)
for bar, val in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val*100:.1f}%', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# STEP 7 — Best Model Evaluation
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, y_pred_best,
      target_names=['Rejected', 'Approved']))

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Rejected', 'Approved'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix — Loan Approval')
plt.tight_layout()
plt.show()

# STEP 8 — Feature Importance
feature_names = list(data.drop('Loan_Status', axis=1).columns)
importances   = best_model.feature_importances_
indices       = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(feature_names)),
        importances[indices],
        color='steelblue')
plt.xticks(range(len(feature_names)),
           [feature_names[i] for i in indices],
           rotation=45, ha='right')
plt.title('Feature Importances — Random Forest')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

# STEP 9 — Predict New Applicant
print("\n--- Predict New Loan Applicant ---")
new_applicant = np.array([[1, 1, 0, 1, 8000, 2000, 150, 1, 2]])
new_scaled    = scaler.transform(new_applicant)
prediction    = best_model.predict(new_scaled)[0]
print("Loan Decision:", "APPROVED ✅" if prediction == 1 else "REJECTED ❌")
