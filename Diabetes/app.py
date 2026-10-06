import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import base64


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "diabetes_model.pkl"
FEATURES_PATH = BASE_DIR / "diabetes_features.pkl"
BACKGROUND_PATH = BASE_DIR / "background.jpg"


# ============================================================
# BACKGROUND IMAGE
# ============================================================

def get_background_image():
    """Convert background image to Base64 so Streamlit can display it reliably."""
    if not BACKGROUND_PATH.exists():
        return None

    try:
        with open(BACKGROUND_PATH, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return None


background_base64 = get_background_image()

if background_base64:
    background_css = f"""
    .stApp {{
        background-image:
            linear-gradient(
                rgba(255, 255, 255, 0.82),
                rgba(255, 255, 255, 0.82)
            ),
            url("data:image/jpeg;base64,{background_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """
else:
    background_css = """
    .stApp {
        background: #f4f7fb;
    }
    """


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
<style>

{background_css}

/* ================= HEADER ================= */

.header {{
    background: linear-gradient(135deg, #087f8c, #00a896);
    padding: 35px 40px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0, 128, 128, 0.18);
}}

.header h1 {{
    color: white;
    font-size: 38px;
    margin: 0;
    font-weight: 750;
}}

.header p {{
    color: #e6fffb;
    font-size: 17px;
    margin-top: 10px;
}}


/* ================= CARDS ================= */

.card {{
    background: rgba(255, 255, 255, 0.96);
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.08);
    margin-bottom: 20px;
}}

.card-title {{
    color: #087f8c;
    font-size: 22px;
    font-weight: 700;
}}

.card-text {{
    color: #64748b;
    font-size: 15px;
    margin-top: 7px;
}}


/* ================= RESULT ================= */

.result-green {{
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border-left: 6px solid #059669;
    padding: 25px;
    border-radius: 15px;
    margin-top: 20px;
}}

.result-red {{
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border-left: 6px solid #dc2626;
    padding: 25px;
    border-radius: 15px;
    margin-top: 20px;
}}

.result-title {{
    font-size: 25px;
    font-weight: 750;
}}

.result-description {{
    color: #475569;
    margin-top: 8px;
}}


/* ================= BUTTON ================= */

.stButton > button {{
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #087f8c, #00a896);
    color: white;
    font-size: 16px;
    font-weight: 700;
}}

.stButton > button:hover {{
    background: linear-gradient(135deg, #066b75, #008f80);
    color: white;
}}


/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {{
    background-color: rgba(255, 255, 255, 0.97);
}}


/* ================= FOOTER ================= */

.footer {{
    text-align: center;
    color: #64748b;
    padding: 35px 0 10px 0;
    font-size: 13px;
}}


/* ================= METRICS ================= */

div[data-testid="stMetric"] {{
    background: rgba(255, 255, 255, 0.96);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
}}

/* ================= INPUTS ================= */

div[data-baseweb="input"] {{
    background: rgba(255, 255, 255, 0.96);
}}

/* ================= DATAFRAME ================= */

div[data-testid="stDataFrame"] {{
    background: rgba(255, 255, 255, 0.96);
    border-radius: 12px;
}}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not MODEL_PATH.exists():
    st.error("❌ diabetes_model.pkl was not found.")
    st.info(
        f"Please place diabetes_model.pkl here:\n\n{MODEL_PATH}"
    )
    st.stop()


if not FEATURES_PATH.exists():
    st.error("❌ diabetes_features.pkl was not found.")
    st.info(
        f"Please place diabetes_features.pkl here:\n\n{FEATURES_PATH}"
    )
    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


# ============================================================
# LOAD FEATURES
# ============================================================

@st.cache_resource
def load_features():
    with open(FEATURES_PATH, "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
    saved_features = load_features()

except Exception as error:
    st.error("❌ Error loading model files.")
    st.exception(error)
    st.stop()


# ============================================================
# FEATURE NAMES
# ============================================================

default_features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]


def get_feature_names(features):
    if isinstance(features, pd.DataFrame):
        return list(features.columns)

    if isinstance(features, pd.Series):
        return list(features.values)

    if isinstance(features, (list, tuple)):
        return list(features)

    if isinstance(features, dict):
        return list(features.keys())

    return default_features


feature_names = get_feature_names(saved_features)

if len(feature_names) != 8:
    feature_names = default_features


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="header">
    <h1>🩺 Diabetes Risk Prediction</h1>
    <p>
        Machine Learning powered diabetes risk assessment
        using a Decision Tree classification model.
    </p>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🩺 Diabetes AI")

    st.markdown("---")

    st.markdown("### About the Application")

    st.write(
        """
        This application uses a trained Machine Learning
        Decision Tree classifier to generate a diabetes
        prediction from eight patient measurements.
        """
    )

    st.markdown("---")

    st.markdown("### 🌳 Machine Learning Model")

    st.success("Decision Tree Classifier")

    st.markdown("---")

    st.markdown("### 📊 Input Features")

    st.write(
        """
        • Pregnancies  
        • Glucose  
        • Blood Pressure  
        • Skin Thickness  
        • Insulin  
        • BMI  
        • Diabetes Pedigree  
        • Age
        """
    )

    st.markdown("---")

    st.caption(
        "Educational ML application — not a medical diagnosis."
    )


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown(
    """
<div class="card">
    <div class="card-title">👤 Patient Information</div>
    <div class="card-text">
        Enter the patient's health measurements to generate a prediction.
    </div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# INPUTS
# ============================================================

left, right = st.columns(2)


with left:

    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )


with right:

    insulin = st.number_input(
        "Insulin Level",
        min_value=0.0,
        max_value=900.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1,
        format="%.2f"
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47,
        step=0.01,
        format="%.2f"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_values = [
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]


input_data = pd.DataFrame(
    [input_values],
    columns=feature_names
)


# ============================================================
# PATIENT SUMMARY
# ============================================================

st.markdown("### 📋 Patient Summary")

m1, m2, m3, m4 = st.columns(4)


with m1:
    st.metric("Glucose", f"{glucose:.0f}")


with m2:
    st.metric("Blood Pressure", f"{blood_pressure:.0f}")


with m3:
    st.metric("BMI", f"{bmi:.1f}")


with m4:
    st.metric("Age", f"{age}")


# ============================================================
# INPUT GRAPH
# ============================================================

st.markdown("### 📊 Patient Measurements")

chart_names = [
    "Glucose",
    "Blood Pressure",
    "Skin Thickness",
    "Insulin",
    "BMI"
]


chart_values = [
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi
]


chart_data = pd.DataFrame({
    "Measurement": chart_names,
    "Value": chart_values
})


fig = px.bar(
    chart_data,
    x="Measurement",
    y="Value",
    color="Measurement",
    color_discrete_sequence=[
        "#087f8c",
        "#00a896",
        "#55c2b4",
        "#f4a261",
        "#e76f51"
    ],
    text="Value"
)


fig.update_traces(
    texttemplate="%{text:.1f}",
    textposition="outside"
)


fig.update_layout(
    height=400,
    showlegend=False,
    plot_bgcolor="white",
    paper_bgcolor="rgba(255,255,255,0.92)",
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Health Measurement",
    yaxis_title="Value"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# ENTERED DATA
# ============================================================

with st.expander("🔍 View Complete Patient Data"):

    summary_data = pd.DataFrame({
        "Feature": feature_names,
        "Value": input_values
    })


    st.dataframe(
        summary_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("")

predict = st.button(
    "🔎  ASSESS DIABETES RISK"
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    try:

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            if hasattr(model, "classes_"):

                classes = list(model.classes_)

                if 1 in classes:

                    diabetes_index = classes.index(1)

                    probability = (
                        probabilities[diabetes_index] * 100
                    )

            elif len(probabilities) >= 2:

                probability = probabilities[1] * 100


        if probability is not None:

            probability = max(
                0,
                min(100, probability)
            )


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.markdown("## 📊 Prediction Result")


        if int(prediction) == 1:

            st.markdown(
                """
                <div class="result-red">
                    <div class="result-title">
                        ⚠️ Diabetes Risk Indicated
                    </div>
                    <div class="result-description">
                        The Decision Tree model predicted an outcome
                        associated with diabetes.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-green">
                    <div class="result-title">
                        ✅ No Diabetes Risk Indicated
                    </div>
                    <div class="result-description">
                        The Decision Tree model predicted an outcome
                        not associated with diabetes.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # PROBABILITY VISUALIZATION
        # ====================================================

        if probability is not None:

            st.markdown("### 🎯 Prediction Probability")

            probability_col, gauge_col = st.columns(2)


            # ------------------------------------------------
            # METRIC
            # ------------------------------------------------

            with probability_col:

                st.metric(
                    "Model Estimated Probability",
                    f"{probability:.1f}%"
                )

                st.progress(
                    int(probability)
                )


            # ------------------------------------------------
            # GAUGE
            # ------------------------------------------------

            with gauge_col:

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability,
                        number={
                            "suffix": "%",
                            "font": {
                                "size": 28
                            }
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100]
                            },
                            "bar": {
                                "color": "#087f8c"
                            },
                            "steps": [
                                {
                                    "range": [0, 30],
                                    "color": "#d1fae5"
                                },
                                {
                                    "range": [30, 70],
                                    "color": "#fef3c7"
                                },
                                {
                                    "range": [70, 100],
                                    "color": "#fee2e2"
                                }
                            ]
                        }
                    )
                )


                gauge.update_layout(
                    height=280,
                    margin=dict(
                        l=20,
                        r=20,
                        t=30,
                        b=10
                    ),
                    paper_bgcolor="rgba(255,255,255,0.0)"
                )


                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.markdown("### 🌳 Model Information")

        model_col1, model_col2, model_col3 = st.columns(3)


        with model_col1:

            st.metric(
                "Algorithm",
                "Decision Tree"
            )


        with model_col2:

            st.metric(
                "Features Used",
                "8"
            )


        with model_col3:

            if hasattr(model, "get_depth"):

                st.metric(
                    "Tree Depth",
                    model.get_depth()
                )

            else:

                st.metric(
                    "Tree Depth",
                    "N/A"
                )


        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        if hasattr(model, "feature_importances_"):

            st.markdown("### 🔬 Feature Importance")

            importance = model.feature_importances_


            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            })


            importance_df = importance_df.sort_values(
                "Importance",
                ascending=True
            )


            importance_fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=[
                    "#d1fae5",
                    "#00a896",
                    "#087f8c"
                ]
            )


            importance_fig.update_layout(
                height=400,
                plot_bgcolor="white",
                paper_bgcolor="rgba(255,255,255,0.92)",
                coloraxis_showscale=False,
                margin=dict(
                    l=20,
                    r=20,
                    t=30,
                    b=20
                ),
                xaxis_title="Importance",
                yaxis_title="Feature"
            )


            st.plotly_chart(
                importance_fig,
                use_container_width=True
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown("---")

        st.warning(
            """
            ⚠️ **Important:** This application is an educational
            machine learning demonstration. The prediction and
            probability shown by the model should not be interpreted
            as a medical diagnosis. Consult a qualified healthcare
            professional for medical advice.
            """
        )


    except Exception as error:

        st.error(
            "❌ Prediction could not be generated."
        )

        st.write(
            "Please verify that the model's training features "
            "and feature order match the application."
        )

        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🩺 Diabetes Risk Prediction System<br>
        Decision Tree Machine Learning • Python • Streamlit • Plotly
    </div>
    """,
    unsafe_allow_html=True
)
