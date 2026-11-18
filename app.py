
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Water Quality Assessment",
    page_icon="💧",
    layout="wide"
)

# ---------------------------------
# LOAD TRAINED MODEL
# ---------------------------------

MODEL_PATH = "water_quality_model.pkl"


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. "
            "Please upload the trained model file."
        )

    data = joblib.load(MODEL_PATH)

    if not isinstance(data, dict):
        raise ValueError(
            "Invalid model file. Expected a dictionary."
        )

    # Support alternative key names from older model files
    aliases = {
        "best_model": ["best_model", "model"],
        "best_model_name": ["best_model_name", "model_name"],
        "imputer": ["imputer"],
        "scaler": ["scaler"],
        "features": ["features"]
    }

    loaded = {}

    for required_key, possible_keys in aliases.items():

        for key in possible_keys:

            if key in data:
                loaded[required_key] = data[key]
                break

        if required_key not in loaded:
            raise KeyError(
                f"Missing key: {required_key}. "
                f"Available keys: {list(data.keys())}"
            )

    return loaded


try:

    data = load_model()

    best_model = data["best_model"]
    best_model_name = data["best_model_name"]
    imputer = data["imputer"]
    scaler = data["scaler"]
    features = data["features"]

except Exception as e:

    st.error(f"Model loading failed: {e}")

    st.info(
        "Check that water_quality_model.pkl "
        "contains the trained model, imputer, scaler "
        "and features."
    )

    st.stop()


# ---------------------------------
# BACKGROUND AND CUSTOM CSS
# ---------------------------------

st.markdown("""
<style>

.stApp {
    background:
    linear-gradient(
        rgba(0, 25, 55, 0.78),
        rgba(0, 35, 65, 0.88)
    ),
    url("https://images.unsplash.com/photo-1500375592092-40eb2168fd21?auto=format&fit=crop&w=2000&q=80");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.hero {
    text-align: center;
    padding: 35px 10px;
    color: white;
}

.hero h1 {
    color: white;
    font-size: 38px;
    font-weight: bold;
}

.hero p {
    color: #d9f5ff;
    font-size: 18px;
}

.section-heading {
    color: white;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    padding: 15px;
}

.info-card {
    background: rgba(255,255,255,0.94);
    padding: 22px;
    border-radius: 15px;
    margin: 12px 0;
    color: #12334a;
}

.footer {
    text-align: center;
    color: white;
    padding: 25px;
    margin-top: 30px;
}

label {
    color: white !important;
    font-weight: bold !important;
}

input {
    color: #12334a !important;
}

.stButton button {
    background-color: #00a6c7;
    color: white;
    font-weight: bold;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------
# HOME PAGE
# ---------------------------------

st.markdown("""
<div class="hero">

<h1>💧 WATER QUALITY ASSESSMENT</h1>

<p>
PREDICTIVE WATER QUALITY ASSESSMENT
FOR POLLUTION AND MANAGEMENT
</p>

<p>
Machine Learning Based Water Pollution
Detection and Potability Prediction
</p>

</div>
""", unsafe_allow_html=True)


# ---------------------------------
# MODEL INFORMATION
# ---------------------------------

st.markdown(
    f"""
    <div class="info-card">
    <h3>🤖 Machine Learning Model</h3>
    <h4>{best_model_name}</h4>
    <p>Selected trained model for water quality prediction.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------
# INPUT PARAMETERS
# ---------------------------------

st.markdown(
    '<div class="section-heading">'
    '🌊 WATER QUALITY PREDICTION SYSTEM'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the physical and chemical parameters "
    "of the water sample below."
)


with st.form("water_quality_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🧪 Basic Water Parameters")

        ph = st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=7.2
        )

        Hardness = st.number_input(
            "Hardness",
            min_value=0.0,
            value=200.0
        )

        Solids = st.number_input(
            "Total Dissolved Solids",
            min_value=0.0,
            value=15000.0
        )

        Chloramines = st.number_input(
            "Chloramines",
            min_value=0.0,
            value=7.0
        )

        Sulfate = st.number_input(
            "Sulfate",
            min_value=0.0,
            value=300.0
        )

    with col2:

        st.subheader("🔬 Additional Parameters")

        Conductivity = st.number_input(
            "Conductivity",
            min_value=0.0,
            value=400.0
        )

        Organic_carbon = st.number_input(
            "Organic Carbon",
            min_value=0.0,
            value=10.0
        )

        Trihalomethanes = st.number_input(
            "Trihalomethanes",
            min_value=0.0,
            value=60.0
        )

        Turbidity = st.number_input(
            "Turbidity",
            min_value=0.0,
            value=3.0
        )

    submitted = st.form_submit_button(
        "🔍 PREDICT WATER QUALITY",
        use_container_width=True
    )


# ---------------------------------
# PREDICTION
# ---------------------------------

if submitted:

    input_data = pd.DataFrame([{
        "ph": ph,
        "Hardness": Hardness,
        "Solids": Solids,
        "Chloramines": Chloramines,
        "Sulfate": Sulfate,
        "Conductivity": Conductivity,
        "Organic_carbon": Organic_carbon,
        "Trihalomethanes": Trihalomethanes,
        "Turbidity": Turbidity
    }])

    try:

        # Check feature compatibility

        if isinstance(features, (list, tuple, np.ndarray)):
            input_data = input_data[list(features)]

        else:
            raise ValueError(
                "Invalid features stored in model file."
            )

        # Apply trained imputer

        processed_data = imputer.transform(input_data)

        # Apply scaler when available

        if scaler is not None:

            processed_data = scaler.transform(
                processed_data
            )

        # Prediction

        prediction = best_model.predict(
            processed_data
        )[0]

        # Class probabilities

        if hasattr(best_model, "predict_proba"):

            probabilities = best_model.predict_proba(
                processed_data
            )[0]

            class_probabilities = dict(
                zip(best_model.classes_, probabilities)
            )

            potable_prob = (
                class_probabilities.get(1, 0) * 100
            )

            non_potable_prob = (
                class_probabilities.get(0, 0) * 100
            )

            confidence = max(probabilities) * 100

        else:

            potable_prob = None
            non_potable_prob = None
            confidence = None


        # ---------------------------------
        # DISPLAY RESULTS
        # ---------------------------------

        st.markdown("---")

        st.markdown(
            '<div class="section-heading">'
            '📊 PREDICTION RESULTS'
            '</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:

            st.success(
                "💧 POTABLE WATER (Predicted)"
            )

            st.write(
                "The model predicts that this sample "
                "belongs to the potable class."
            )

        else:

            st.error(
                "⚠️ NON-POTABLE WATER (Predicted)"
            )

            st.write(
                "The model predicts that this sample "
                "belongs to the non-potable class."
            )


        # ---------------------------------
        # RESULT METRICS
        # ---------------------------------

        if confidence is not None:

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

            with c2:

                st.metric(
                    "Potable Probability",
                    f"{potable_prob:.2f}%"
                )


            # ---------------------------------
            # PROBABILITY CHART
            # ---------------------------------

            probability_data = pd.DataFrame({

                "Water Class": [
                    "Potable",
                    "Non-Potable"
                ],

                "Probability (%)": [
                    potable_prob,
                    non_potable_prob
                ]

            })

            st.subheader("📈 Class Probability")

            st.bar_chart(
                probability_data.set_index(
                    "Water Class"
                )
            )

        else:

            st.warning(
                "Probability and confidence are unavailable "
                "because this model does not support "
                "predict_proba()."
            )


        st.info(
            f"Selected Model: {best_model_name}"
        )


    except Exception as e:

        st.error(
            f"Prediction error: {str(e)}"
        )


# ---------------------------------
# ABOUT PROJECT
# ---------------------------------

st.markdown("---")

st.markdown("""
<div class="info-card">

<h2>🌍 About This Project</h2>

<p>
This project predicts water potability using machine learning.
It uses physical and chemical water quality parameters to
classify water samples into potable and non-potable classes.
</p>

<h3>Technologies Used</h3>

<ul>
<li>Python</li>
<li>Machine Learning</li>
<li>Scikit-learn</li>
<li>Pandas and NumPy</li>
<li>Streamlit</li>
</ul>

</div>
""", unsafe_allow_html=True)


# ---------------------------------
# FOOTER
# ---------------------------------

st.markdown("""
<div class="footer">

<h3>💧 Predictive Water Quality Assessment</h3>

<p>
Pollution Detection and Management Using Machine Learning
</p>

<p>
Note: This prediction is not a substitute for
certified laboratory water testing.
</p>

</div>
""", unsafe_allow_html=True)
