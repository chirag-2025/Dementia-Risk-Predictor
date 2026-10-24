# Predicting Dementia Using Brain Scan Measurements and Patient Health Data

Healthcare Analytics mini project (CA-III), Symbiosis Institute of Technology, Nagpur.
Chirag Somkuwar, PRN 23070521042.

## What this project does

Predicts whether a person belongs to the demented or nondemented group by combining two kinds of data:

- brain scan measurements: eTIV, nWBV, ASF
- patient details: age, sex, SES, MMSE

A Random Forest is trained on OASIS-2 and tested on OASIS-1. SHAP is used on the combined model to see whether the brain scan measurements or the clinical details influence the prediction more.

## Dataset

OASIS-1 (cross-sectional) and OASIS-2 (longitudinal), from https://www.oasis-brains.org/

The data files are not included in this repository because OASIS has its own access terms. Download them from the OASIS site. For the analysis in this repo, I used a copy of the OASIS CSV files from Kaggle (jboysen/mri-and-alzheimers) while my access request was pending.

## Files

- `oasis_eda_preprocessing.py`: exploration and cleaning
- `oasis_model_building.py`: Logistic Regression, Random Forest and SVM compared on a held-out split
- `oasis_final_model.py`: imaging vs clinical vs combined, SHAP over 5 splits, saves `model.pkl`
- `oasis1_external_test.py`: trained on OASIS-2, tested on OASIS-1
- `app.py`: Streamlit dashboard (needs `model.pkl` and `requirements.txt`)

## Run the dashboard

The dashboard uses the included model bundle and does not need the original
OASIS CSV files to run. In PowerShell, from the project folder, run:

```
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The project virtual environment should use Python 3.12. Streamlit Cloud should
also be configured to use Python 3.12 and deploy `app.py` from the `main`
branch. Prediction inputs are submitted together with **Update estimate** to
avoid running the explanation for every slider movement.

## Notes

The splits keep all visits of one person together, so the same person is never in both train and test. This is a student project for learning and is not a medical tool.
