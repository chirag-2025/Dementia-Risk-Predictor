from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Dementia Risk Predictor",
    page_icon=":material/neurology:",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Dementia Risk Predictor")
st.caption("Healthcare Analytics · CA-III · Educational prototype")
st.write(
    "Explore how brain-scan measurements and clinical details influence a "
    "machine-learning estimate for the OASIS study groups."
)
st.info(
    "For learning only. This prototype is not a medical tool and must not be "
    "used for diagnosis or treatment decisions.",
    icon=":material/info:",
)

loading_message = st.empty()
loading_message.info("Loading the prediction model...")


@st.cache_resource
def load_model():
    model_path = BASE_DIR / "model.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Required model file was not found: {model_path}")
    bundle = joblib.load(model_path)
    explainer = shap.TreeExplainer(bundle["model"])
    return bundle, explainer


try:
    bundle, explainer = load_model()
except FileNotFoundError as error:
    loading_message.error(str(error))
    st.stop()
else:
    loading_message.empty()

model = bundle["model"]
features = tuple(bundle["features"])
imaging = tuple(bundle["imaging"])
clinical = tuple(bundle["clinical"])
ranges = bundle["ranges"]

NAMES = {
    "eTIV": "eTIV",
    "nWBV": "nWBV",
    "ASF": "ASF",
    "Age": "Age",
    "SES": "SES",
    "MMSE": "MMSE",
    "M/F": "Sex",
}

EXAMPLES = {
    "Typical nondemented person": {
        "etiv": 1474,
        "nwbv": 0.739,
        "age": 77,
        "sex": "Female",
        "ses": 2,
        "mmse": 29,
    },
    "Typical demented person": {
        "etiv": 1463,
        "nwbv": 0.713,
        "age": 76,
        "sex": "Male",
        "ses": 2,
        "mmse": 27,
    },
    "Low MMSE, small brain volume": {
        "etiv": 1480,
        "nwbv": 0.68,
        "age": 80,
        "sex": "Male",
        "ses": 3,
        "mmse": 18,
    },
}
START = {
    "etiv": 1469,
    "nwbv": 0.726,
    "age": 76,
    "sex": "Female",
    "ses": 2,
    "mmse": 28,
}

ETIV_LO, ETIV_HI = int(ranges["eTIV"][0]), int(ranges["eTIV"][1])
NWBV_LO, NWBV_HI = round(ranges["nWBV"][0], 2), round(ranges["nWBV"][1], 2)
AGE_LO, AGE_HI = int(ranges["Age"][0]), int(ranges["Age"][1])

for key, value in START.items():
    st.session_state.setdefault(key, value)


def use_example(name):
    for key, value in EXAMPLES[name].items():
        st.session_state[key] = value


@st.cache_data(max_entries=128, show_spinner=False)
def predict_and_explain(values, feature_names, _model, _explainer):
    row = pd.DataFrame([values], columns=feature_names)
    probability = float(_model.predict_proba(row)[0, 1])
    shap_values = _explainer.shap_values(row)
    if isinstance(shap_values, list):
        contributions = shap_values[1][0]
    elif shap_values.ndim == 3:
        contributions = shap_values[0, :, 1]
    else:
        contributions = shap_values[0]
    return probability, tuple(float(value) for value in contributions)


@st.cache_data(max_entries=128, show_spinner=False)
def calculate_what_if(values, feature_names, varied_feature, nwbv_range, _model):
    row = pd.DataFrame([values], columns=feature_names)
    if varied_feature == "MMSE":
        grid = np.arange(0, 31)
    else:
        grid = np.linspace(nwbv_range[0], nwbv_range[1], 60)
    scenarios = pd.concat([row] * len(grid), ignore_index=True)
    scenarios[varied_feature] = grid
    scores = _model.predict_proba(scenarios[list(feature_names)])[:, 1] * 100
    return tuple(float(value) for value in grid), tuple(float(value) for value in scores)


@st.cache_data(ttl=3600, max_entries=8)
def load_results_csv(path):
    return pd.read_csv(path)


view = st.segmented_control(
    "Dashboard section",
    ["Predict", "How it works", "Results and limits"],
    default="Predict",
    key="dashboard_view",
    width="stretch",
)

if view == "Predict":
    st.subheader("Prediction")
    st.caption("Choose an example, adjust the inputs, then update the estimate.")
    with st.container(horizontal=True, gap="small"):
        for name in EXAMPLES:
            st.button(
                name,
                key=f"example_{name}",
                on_click=use_example,
                args=(name,),
                width="stretch",
                help="Load illustrative sample values into the input form.",
            )

    with st.form("prediction_inputs", border=True):
        scan_column, clinical_column = st.columns(2, gap="large")
        with scan_column:
            st.markdown("**Brain scan measurements**")
            st.slider(
                "eTIV (estimated total intracranial volume)",
                ETIV_LO,
                ETIV_HI,
                key="etiv",
                help="Estimated total intracranial volume from the scan.",
            )
            st.slider(
                "nWBV (normalized whole brain volume)",
                NWBV_LO,
                NWBV_HI,
                step=0.001,
                format="%.3f",
                key="nwbv",
                help="Normalized whole brain volume from the scan.",
            )
            asf = bundle["asf_const"] / st.session_state["etiv"]
            st.caption(f"Derived ASF: {asf:.3f}")

        with clinical_column:
            st.markdown("**Patient details**")
            st.slider("Age", AGE_LO, AGE_HI, key="age")
            st.selectbox("Sex", ["Female", "Male"], key="sex")
            st.slider(
                "Socioeconomic status (1 = highest, 5 = lowest)",
                1,
                5,
                key="ses",
            )
            st.slider("MMSE score (out of 30)", 0, 30, key="mmse")

        st.form_submit_button(
            "Update estimate",
            type="primary",
            width="stretch",
            icon=":material/analytics:",
        )

    state = st.session_state
    asf = bundle["asf_const"] / state["etiv"]
    input_values = {
        "eTIV": state["etiv"],
        "nWBV": state["nwbv"],
        "ASF": asf,
        "Age": state["age"],
        "SES": state["ses"],
        "MMSE": state["mmse"],
        "M/F": 1 if state["sex"] == "Male" else 0,
    }
    row_values = tuple(input_values[feature] for feature in features)
    probability, contribution_values = predict_and_explain(
        row_values, features, model, explainer
    )
    contributions = pd.Series(contribution_values, index=features)

    st.subheader("Estimate")
    with st.container(horizontal=True, gap="small"):
        st.metric(
            "Dementia-group model score",
            f"{probability * 100:.0f}%",
            border=True,
            help="The score is not calibrated as an individual clinical probability.",
        )
        st.metric(
            "Model classification",
            "Demented" if probability >= 0.5 else "Nondemented",
            border=True,
        )
    st.progress(probability)
    st.caption(
        "This model score is not a diagnosis or a calibrated clinical probability."
    )
    if 0.4 <= probability <= 0.6:
        st.info("The score is close to the decision threshold; the model is uncertain.")

    st.subheader("Factors influencing this estimate")
    ordered_features = contributions.abs().sort_values().index
    labels = []
    for feature in ordered_features:
        value = input_values[feature]
        if feature == "M/F":
            shown = state["sex"]
        elif feature in ("nWBV", "ASF"):
            shown = f"{value:.3f}"
        else:
            shown = f"{value:.0f}"
        labels.append(f"{NAMES[feature]} = {shown}")

    figure, axis = plt.subplots(figsize=(7, 4))
    bar_colors = [
        "#c2410c" if contributions[feature] > 0 else "#1d4ed8"
        for feature in ordered_features
    ]
    axis.barh(labels, [contributions[feature] for feature in ordered_features], color=bar_colors)
    axis.axvline(0, color="#64748b", linewidth=0.8)
    axis.set_xlabel("Contribution toward nondemented ←  |  → demented")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    st.pyplot(
        figure,
        width="stretch",
        alt="Horizontal SHAP chart showing how each input contributed to this estimate.",
    )
    plt.close(figure)

    imaging_push = contributions[list(imaging)].sum()
    clinical_push = contributions[list(clinical)].sum()

    def direction(value):
        return "toward demented" if value > 0 else "toward nondemented"

    with st.container(horizontal=True, gap="small"):
        st.metric(
            "Imaging contribution",
            f"{imaging_push:+.2f}",
            help=f"Combined influence from eTIV, nWBV, and ASF: {direction(imaging_push)}.",
            border=True,
        )
        st.metric(
            "Clinical contribution",
            f"{clinical_push:+.2f}",
            help=f"Combined influence from age, sex, SES, and MMSE: {direction(clinical_push)}.",
            border=True,
        )

    st.subheader("Explore one input")
    varied_feature = st.segmented_control(
        "Input to vary",
        ["MMSE", "nWBV"],
        default="MMSE",
        key="what_if_feature",
        help="All other entered values remain unchanged.",
    )
    grid, scores = calculate_what_if(
        row_values,
        features,
        varied_feature,
        (NWBV_LO, NWBV_HI),
        model,
    )
    what_if_data = pd.DataFrame(
        {varied_feature: grid, "Dementia-group model score (%)": scores}
    )
    st.line_chart(
        what_if_data,
        x=varied_feature,
        y="Dementia-group model score (%)",
        y_label="Model score (%)",
        alt=f"Model score as {varied_feature} changes while other inputs remain fixed.",
    )
    st.caption("The chart shows model behavior, not a causal effect.")

elif view == "How it works":
    st.subheader("How the estimate is produced")
    st.write(
        "A Random Forest trained on OASIS-2 combines three scan measurements "
        "(eTIV, nWBV, and ASF) with age, sex, socioeconomic status, and MMSE. "
        "SHAP summarizes how the fitted model used those inputs for an estimate."
    )
    st.subheader("Relative influence by input group")
    st.write(
        "Across five analysis splits, imaging accounts for about 34% of mean "
        "absolute SHAP influence and clinical details about 66%."
    )
    modality_figure = BASE_DIR / "modality_share_final.png"
    if modality_figure.is_file():
        st.image(
            modality_figure,
            caption="Imaging versus clinical share of SHAP influence",
            alt="Bar chart showing approximately 34 percent imaging and 66 percent clinical influence.",
            width="stretch",
        )
    feature_figure = BASE_DIR / "shap_summary_final.png"
    if feature_figure.is_file():
        st.subheader("Feature-level influence")
        st.image(
            feature_figure,
            caption="SHAP summary; each dot represents a visit",
            alt="SHAP summary plot showing feature values and their influence on model scores.",
            width="stretch",
        )
    st.caption(
        "An explanation describes model behavior. It does not show what causes dementia."
    )

elif view == "Results and limits":
    st.subheader("Evaluation results")
    st.write(
        "OASIS-2 contains 150 people and 373 visits. All visits from each person "
        "were kept together during cross-validation. The OASIS-1 results below "
        "are a separate external test."
    )
    result_tables = (
        (
            "Cross-validation on OASIS-2: modality analysis",
            "final_model_summary.csv",
        ),
        (
            "Cross-validation on OASIS-2: classifier comparison",
            "model_comparison_results.csv",
        ),
        ("External test: trained on OASIS-2, tested on OASIS-1", "oasis1_test_results.csv"),
    )
    for title, filename in result_tables:
        csv_path = BASE_DIR / filename
        if csv_path.is_file():
            st.markdown(f"**{title}**")
            st.dataframe(
                load_results_csv(str(csv_path)),
                hide_index=True,
                width="stretch",
                alt=title,
            )
    st.subheader("Limitations")
    st.write(
        "The dataset is small and limited to adults aged 60 and older. MMSE is "
        "closely related to how the dementia group is defined, and the model "
        "score is not a calibrated probability. The final dashboard model is "
        "trained on all OASIS-2 records and has no prospective clinical "
        "validation. SHAP explains the model, not the causes of dementia."
    )
