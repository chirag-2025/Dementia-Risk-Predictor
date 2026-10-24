import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Dementia Risk Predictor", layout="centered")

st.title("Dementia Risk Predictor")
st.write(
    "A small course project (Healthcare Analytics, CA-III). Enter a person's "
    "brain scan measurements and basic clinical details, and the model gives "
    "an estimate of whether they belong to the demented group, along with "
    "which inputs pushed the estimate up or down."
)
st.caption("This is for learning only. It is not a medical tool and should "
           "never be used to diagnose anyone.")

loading_message = st.empty()
loading_message.info("Loading the prediction model...")


@st.cache_resource
def load_model():
    bundle = joblib.load("model.pkl")
    explainer = shap.TreeExplainer(bundle["model"])
    return bundle, explainer


bundle, explainer = load_model()
loading_message.empty()
model = bundle["model"]
features = bundle["features"]
imaging = bundle["imaging"]
clinical = bundle["clinical"]
ranges = bundle["ranges"]

NAMES = {"eTIV": "eTIV", "nWBV": "nWBV", "ASF": "ASF", "Age": "Age",
         "SES": "SES", "MMSE": "MMSE", "M/F": "Sex"}

# ---------------- example people ----------------
# The first two are the median values of each group in OASIS-2.
# The third is a made-up example with low scores, to show the other end.
EXAMPLES = {
    "Typical nondemented person": dict(etiv=1474, nwbv=0.739, age=77, sex="Female", ses=2, mmse=29),
    "Typical demented person": dict(etiv=1463, nwbv=0.713, age=76, sex="Male", ses=2, mmse=27),
    "Low MMSE, small brain volume": dict(etiv=1480, nwbv=0.68, age=80, sex="Male", ses=3, mmse=18),
}
START = dict(etiv=1469, nwbv=0.726, age=76, sex="Female", ses=2, mmse=28)

ETIV_LO, ETIV_HI = int(ranges["eTIV"][0]), int(ranges["eTIV"][1])
NWBV_LO, NWBV_HI = round(ranges["nWBV"][0], 2), round(ranges["nWBV"][1], 2)
AGE_LO, AGE_HI = int(ranges["Age"][0]), int(ranges["Age"][1])

for key, value in START.items():
    st.session_state.setdefault(key, value)


def use_example(name):
    # copy the example's values into the sliders
    for key, value in EXAMPLES[name].items():
        st.session_state[key] = value


tab1, tab2, tab3 = st.tabs(["Predict", "How the model works", "Results and limits"])

# =====================================================================
# TAB 1: predict
# =====================================================================
with tab1:
    st.write("**Try an example person**")
    cols = st.columns(len(EXAMPLES))
    for col, name in zip(cols, EXAMPLES):
        col.button(name, on_click=use_example, args=(name,), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Brain scan measurements")
        st.slider("eTIV (estimated total intracranial volume)",
                  ETIV_LO, ETIV_HI, key="etiv")
        st.slider("nWBV (normalized whole brain volume)",
                  NWBV_LO, NWBV_HI, step=0.001, format="%.3f", key="nwbv")
        asf = bundle["asf_const"] / st.session_state["etiv"]
        st.caption(f"ASF is worked out from eTIV: {asf:.3f}")
    with right:
        st.subheader("Patient details")
        st.slider("Age", AGE_LO, AGE_HI, key="age")
        st.selectbox("Sex", ["Female", "Male"], key="sex")
        st.slider("Socioeconomic status (1 = highest, 5 = lowest)", 1, 5, key="ses")
        st.slider("MMSE score (out of 30)", 0, 30, key="mmse")

    s = st.session_state
    row = pd.DataFrame([{
        "eTIV": s["etiv"], "nWBV": s["nwbv"], "ASF": asf, "Age": s["age"],
        "SES": s["ses"], "MMSE": s["mmse"], "M/F": 1 if s["sex"] == "Male" else 0,
    }], columns=features)

    # ---------- prediction ----------
    prob = float(model.predict_proba(row)[0, 1])

    st.divider()
    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.metric("Chance of demented group", f"{prob * 100:.0f}%")
    c2.metric("Model's call", "Demented" if prob >= 0.5 else "Nondemented")
    st.progress(prob)
    if 0.4 <= prob <= 0.6:
        st.info("The estimate is close to 50%, so the model is not sure about this one.")

    # ---------- what pushed it (SHAP for this one person) ----------
    sv = explainer.shap_values(row)
    if isinstance(sv, list):
        vals = sv[1][0]
    elif sv.ndim == 3:
        vals = sv[0, :, 1]
    else:
        vals = sv[0]
    contrib = pd.Series(vals, index=features)

    st.subheader("What pushed this estimate")
    order = contrib.abs().sort_values().index
    labels = []
    for f in order:
        v = row.iloc[0][f]
        if f == "M/F":
            shown = s["sex"]
        elif f in ("nWBV", "ASF"):
            shown = f"{v:.3f}"
        else:
            shown = f"{v:.0f}"
        labels.append(f"{NAMES[f]} = {shown}")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    colors = ["#e8431f" if contrib[f] > 0 else "#2563eb" for f in order]
    ax.barh(labels, [contrib[f] for f in order], color=colors)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("Towards Nondemented   |   Towards Demented")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    img_push = contrib[imaging].sum()
    clin_push = contrib[clinical].sum()

    def direction(x):
        return "towards demented" if x > 0 else "towards nondemented"

    st.write(
        f"For this person, the brain scan measurements together pushed the "
        f"estimate **{direction(img_push)}** ({img_push:+.2f}), and the clinical "
        f"details pushed it **{direction(clin_push)}** ({clin_push:+.2f})."
    )

    # ---------- what-if curve ----------
    st.divider()
    st.subheader("What if this one input were different?")
    st.caption("Everything else stays as entered above. Only the chosen input changes.")
    vary = st.selectbox("Input to change", ["MMSE", "nWBV"])
    if vary == "MMSE":
        grid = np.arange(0, 31)
    else:
        grid = np.linspace(NWBV_LO, NWBV_HI, 60)
    what_if = pd.concat([row] * len(grid), ignore_index=True)
    what_if[vary] = grid
    curve = model.predict_proba(what_if[features])[:, 1] * 100

    fig2, ax2 = plt.subplots(figsize=(6, 2.8))
    ax2.plot(grid, curve, color="#2563eb", linewidth=2)
    ax2.axhline(50, color="#999999", linewidth=0.8, linestyle="--")
    ax2.scatter([row.iloc[0][vary]], [prob * 100], color="#e8431f", zorder=3, s=45)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel(vary)
    ax2.set_ylabel("Chance of demented group (%)")
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption("The orange dot is the person entered above. The dashed line is 50%.")

# =====================================================================
# TAB 2: how the model works
# =====================================================================
with tab2:
    st.subheader("The idea")
    st.write(
        "Two kinds of information go into one model: brain scan measurements "
        "(eTIV, nWBV, ASF) and patient details (age, sex, SES, MMSE). The model "
        "is a Random Forest trained on the OASIS-2 data. SHAP is then used to see "
        "how much each input pushes a prediction up or down."
    )
    st.subheader("How much comes from each kind of data?")
    st.write(
        "Adding up the SHAP influence of each group, brain scan measurements "
        "carry about a third (34% on average over 5 splits) and clinical "
        "details about two thirds."
    )
    if Path("modality_share_final.png").exists():
        st.image("modality_share_final.png", caption="Imaging vs clinical share of SHAP influence")
    st.subheader("Which inputs matter most?")
    if Path("shap_summary_final.png").exists():
        st.image("shap_summary_final.png", caption="SHAP summary: each dot is one visit")
    st.write(
        "How to read the plot: each row is an input and each dot is one visit. "
        "Dots on the right pushed the prediction towards demented, dots on the left "
        "towards nondemented. Colour shows whether the input value was high or low. "
        "For example, low MMSE and low nWBV push towards demented."
    )

# =====================================================================
# TAB 3: results and limits
# =====================================================================
with tab3:
    st.subheader("How the model was tested")
    st.write(
        "Random Forest trained on the OASIS-2 longitudinal data (150 people, "
        "373 visits). The label is demented (including people who converted "
        "during the study) versus nondemented. All visits of one person were "
        "kept together when splitting the data."
    )
    s1 = Path("final_model_summary.csv")
    if s1.exists():
        st.write("Cross-validation on OASIS-2, average of 5 seeds:")
        st.dataframe(pd.read_csv(s1), hide_index=True)
    s3 = Path("model_comparison_results.csv")
    if s3.exists():
        st.write("Three models compared on OASIS-2 (same cross-validation):")
        st.dataframe(pd.read_csv(s3), hide_index=True)
    s2 = Path("oasis1_test_results.csv")
    if s2.exists():
        st.write("Trained on OASIS-2, tested on OASIS-1:")
        st.dataframe(pd.read_csv(s2), hide_index=True)
    st.subheader("Limits")
    st.write(
        "Small dataset and people aged 60 and above only. MMSE is closely linked to "
        "how the dementia label is assigned, so it has the biggest influence. The "
        "models miss about 3 in 10 demented cases on OASIS-2. The percentage is the "
        "model's score, not a true probability. The final model is trained on all the "
        "data, so it can be overconfident, especially for inputs close to the people "
        "it learned from. The explanation chart shows what the model relied on, not "
        "what causes dementia."
    )
