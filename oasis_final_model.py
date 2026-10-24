# OASIS final model (7 features, no education)
# Run in Colab after:  !pip install shap
# Needs oasis_longitudinal_cleaned.csv in the file panel.
# Saves: model.pkl, final_model_summary.csv, shap_summary_final.png, modality_share_final.png

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit, cross_val_score

df = pd.read_csv("oasis_longitudinal_cleaned.csv")

imaging = ['eTIV', 'nWBV', 'ASF']
clinical = ['Age', 'SES', 'MMSE', 'M/F']
all_feats = imaging + clinical
y = df['Group_Binary_Encoded']
groups = df['Subject ID']          # keep one patient's visits together

# ---------- 1. Imaging vs clinical vs combined, repeated with 5 seeds ----------
sets = {"Imaging only": imaging, "Clinical only": clinical, "Combined": all_feats}
rows = []
for name, feats in sets.items():
    seed_means = []
    for seed in range(5):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        rf = RandomForestClassifier(n_estimators=200, random_state=seed)
        s = cross_val_score(rf, df[feats], y, cv=cv, groups=groups, scoring='roc_auc')
        seed_means.append(s.mean())
    rows.append({"Feature set": name,
                 "Mean ROC-AUC": round(np.mean(seed_means), 3),
                 "Lowest seed": round(min(seed_means), 3),
                 "Highest seed": round(max(seed_means), 3)})

summary = pd.DataFrame(rows)
print("=== 5-fold CV grouped by patient, repeated with 5 seeds ===")
print(summary.to_string(index=False))
summary.to_csv("final_model_summary.csv", index=False)

# ---------- 2. SHAP on 5 different patient splits ----------
def dementia_shap(sv):
    if isinstance(sv, list):
        return sv[1]
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv

X = df[all_feats]
shares = []
for seed in range(5):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(X, y, groups))
    rf = RandomForestClassifier(n_estimators=200, random_state=seed).fit(X.iloc[tr], y.iloc[tr])
    sv = dementia_shap(shap.TreeExplainer(rf).shap_values(X.iloc[te]))
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=all_feats)
    shares.append({"seed": seed,
                   "Imaging %": round(mean_abs[imaging].sum() / mean_abs.sum() * 100, 1),
                   "Clinical %": round(mean_abs[clinical].sum() / mean_abs.sum() * 100, 1),
                   "Top feature": mean_abs.idxmax()})
    if seed == 0:
        shap.summary_plot(sv, X.iloc[te], show=False)
        plt.title("SHAP summary (7 features): push towards Demented")
        plt.tight_layout()
        plt.savefig("shap_summary_final.png", dpi=150)
        plt.show()

shares_df = pd.DataFrame(shares)
print("\n=== Imaging vs clinical share of SHAP influence, 5 splits ===")
print(shares_df.to_string(index=False))
print("\nImaging % average:", round(shares_df['Imaging %'].mean(), 1),
      "| range:", shares_df['Imaging %'].min(), "to", shares_df['Imaging %'].max())

avg = [shares_df['Imaging %'].mean(), shares_df['Clinical %'].mean()]
plt.figure(figsize=(5, 4))
plt.bar(["Imaging", "Clinical"], avg, color=["#777777", "#333333"])
plt.ylabel("Average share of SHAP influence (%)")
plt.title("Imaging vs clinical (average of 5 splits)")
plt.tight_layout()
plt.savefig("modality_share_final.png", dpi=150)
plt.show()

# ---------- 3. Final model for the dashboard (trained on all OASIS-2 rows) ----------
final_rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X, y)
joblib.dump({
    "model": final_rf,
    "features": all_feats,
    "imaging": imaging,
    "clinical": clinical,
    # ASF is worked out from eTIV (ASF x eTIV is almost constant), so the
    # dashboard only asks for eTIV and calculates ASF itself
    "asf_const": float((df['ASF'] * df['eTIV']).median()),
    "ranges": {c: [float(df[c].min()), float(df[c].max())]
               for c in ['Age', 'MMSE', 'eTIV', 'nWBV']},
}, "model.pkl")

# requirements.txt for deployment: scikit-learn must match the version that
# built model.pkl, otherwise the saved model may not load on the server
import sklearn
with open("requirements.txt", "w") as f:
    f.write("streamlit\npandas\nnumpy\nmatplotlib\nshap\njoblib\n"
            f"scikit-learn=={sklearn.__version__}\n")

print("\nSaved model.pkl and requirements.txt")
print("Download both from the file panel for the dashboard step.")
