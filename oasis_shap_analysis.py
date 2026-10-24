# OASIS dementia project: modality comparison + SHAP
# Run in the same Colab notebook (needs oasis_longitudinal_cleaned.csv)
# First cell:  !pip install shap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score

df = pd.read_csv("oasis_longitudinal_cleaned.csv")

imaging = ['eTIV', 'nWBV', 'ASF']
clinical = ['Age', 'EDUC', 'SES', 'MMSE', 'M/F']
y = df['Group_Binary_Encoded']

# Same patient has several visits in OASIS-2, so split by Subject ID.
# A normal random split can put one person in both train and test
# and make the scores look better than they really are.
groups = df['Subject ID']

# ---------- Part 1: imaging only vs clinical only vs combined ----------
feature_sets = {
    "Imaging only": imaging,
    "Clinical only": clinical,
    "Combined": imaging + clinical,
}

rows = []
for name, feats in feature_sets.items():
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    scores = cross_val_score(rf, df[feats], y, cv=GroupKFold(n_splits=5),
                             groups=groups, scoring='roc_auc')
    rows.append({"Feature set": name,
                 "Mean ROC-AUC": round(scores.mean(), 3),
                 "Std": round(scores.std(), 3)})

comparison = pd.DataFrame(rows)
print("=== 5-fold CV (grouped by patient) ===")
print(comparison.to_string(index=False))
comparison.to_csv("modality_comparison.csv", index=False)

# ---------- Part 2: SHAP on the combined model ----------
all_feats = imaging + clinical
X = df[all_feats]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf)
sv = explainer.shap_values(X_test)

# different shap versions return different shapes, pick the "Demented" class
if isinstance(sv, list):
    sv_dem = sv[1]
elif sv.ndim == 3:
    sv_dem = sv[:, :, 1]
else:
    sv_dem = sv

shap.summary_plot(sv_dem, X_test, show=False)
plt.title("SHAP summary: which features push towards Demented")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.show()

# ---------- Part 3: imaging vs clinical share of influence ----------
mean_abs = pd.Series(np.abs(sv_dem).mean(axis=0), index=all_feats)
print("\nMean |SHAP| per feature:")
print(mean_abs.sort_values(ascending=False).round(4))

img_share = mean_abs[imaging].sum() / mean_abs.sum() * 100
clin_share = mean_abs[clinical].sum() / mean_abs.sum() * 100
print(f"\nImaging features: {img_share:.1f}% of total influence")
print(f"Clinical features: {clin_share:.1f}% of total influence")

plt.figure(figsize=(5, 4))
plt.bar(["Imaging", "Clinical"], [img_share, clin_share], color=["#777777", "#333333"])
plt.ylabel("Share of total SHAP influence (%)")
plt.title("Imaging vs clinical features")
plt.tight_layout()
plt.savefig("modality_share.png", dpi=150)
plt.show()
