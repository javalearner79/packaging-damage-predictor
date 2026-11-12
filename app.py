"""Streamlit app for packaging damage prediction and operations guidance."""

import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from google import genai


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
GEMINI_MODEL ="gemini-3.8-flash"

FEATURE_COLUMNS = [
    "product_weight_kg",
    "transit_distance_km",
    "material_burst_strength_kpa",
    "fragility_score",
    "cost_inr",
]


st.set_page_config(
    page_title="Packaging Damage & Cost Predictor",
    page_icon="📦",
    layout="centered",
)


@st.cache_resource
def load_model():
    """Load the trained scikit-learn model once per app session."""
    return joblib.load(MODEL_PATH)


def get_gemini_recommendation(inputs: dict, prediction: float, api_key: str) -> str:
    """Ask Gemini for a concise operational recommendation grounded in the inputs."""
    client = genai.Client(api_key=api_key)

    prompt = f"""You are an experienced e-commerce fulfillment and supply-chain
operations consultant. Give a practical recommendation for one parcel using
the supplied packaging characteristics and the model's predicted damage risk.

Parcel inputs:
- Product weight: {inputs['product_weight_kg']:.2f} kg
- Transit distance: {inputs['transit_distance_km']:.0f} km
- Material burst strength: {inputs['material_burst_strength_kpa']:.0f} kPa
- Fragility score: {inputs['fragility_score']} out of 5
- Packaging cost: ₹{inputs['cost_inr']:.2f}
- ML-predicted transit damage probability: {prediction:.1f}%

Recommend whether to accept the current packaging or consider an upgrade.
When useful, discuss stronger or higher-ply corrugate, cushioning, or a
different material, and explain the cost-risk trade-off. Be specific, concise,
and use a short recommendation followed by a few actionable bullets. Do not
claim that a packaging change is guaranteed to prevent damage. Treat the model
prediction as an estimate, not a measured real-world probability."""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config={"temperature": 0.4, "max_output_tokens": 800},
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty recommendation.")
    return response.text


st.title("📦 Packaging Damage & Cost Predictor")
st.caption(
    "Estimate parcel damage risk and get an AI-assisted packaging recommendation."
)

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH.name}")
    st.info("Run `train_model.py` first to create model.pkl in this project folder.")
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load model.pkl: {exc}")
    st.stop()

st.subheader("Parcel and packaging details")
left, right = st.columns(2)
with left:
    product_weight = st.number_input(
        "Product weight (kg)", min_value=0.1, max_value=25.0, value=2.0, step=0.1
    )
    transit_distance = st.number_input(
        "Transit distance (km)", min_value=20, max_value=2500, value=500, step=25
    )
    burst_strength = st.number_input(
        "Material burst strength (kPa)",
        min_value=150,
        max_value=1000,
        value=500,
        step=10,
    )
with right:
    fragility_score = st.slider("Fragility score", min_value=1, max_value=5, value=3)
    packaging_cost = st.number_input(
        "Packaging cost (INR)", min_value=10.0, max_value=2500.0, value=50.0, step=5.0
    )

if st.button("Predict Damage & Optimize", type="primary", use_container_width=True):
    parcel_inputs = {
        "product_weight_kg": float(product_weight),
        "transit_distance_km": float(transit_distance),
        "material_burst_strength_kpa": float(burst_strength),
        "fragility_score": int(fragility_score),
        "cost_inr": float(packaging_cost),
    }
    features = pd.DataFrame([parcel_inputs], columns=FEATURE_COLUMNS)

    try:
        damage_probability = float(model.predict(features)[0])
        damage_probability = max(0.0, min(100.0, damage_probability))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.subheader("Prediction")
    metric_col, status_col = st.columns([1, 2])
    with metric_col:
        st.metric("Transit damage probability", f"{damage_probability:.1f}%")
    with status_col:
        if damage_probability >= 40:
            st.error("High estimated risk — review protective packaging.")
        elif damage_probability >= 20:
            st.warning("Moderate estimated risk — consider a packaging review.")
        else:
            st.success("Lower estimated risk for the entered shipment profile.")

    st.caption(
        "The estimate comes from a Random Forest trained on synthetic portfolio data; "
        "validate it against actual shipment outcomes before operational use."
    )

    api_key = os.getenv("GEMINI_API_KEY")
    st.subheader("AI operations recommendation")
    if not api_key:
        st.warning(
            "Gemini was not called because GEMINI_API_KEY is not set. "
            "Set the key as shown below, then restart Streamlit."
        )
    else:
        with st.spinner("Preparing a packaging recommendation…"):
            try:
                recommendation = get_gemini_recommendation(
                    parcel_inputs, damage_probability, api_key
                )
                st.markdown(recommendation)
            except Exception as exc:
                st.error(f"Gemini recommendation could not be generated: {exc}")

with st.expander("Set up the Gemini API key"):
    st.markdown(
        "Install the app dependencies with `pip install streamlit scikit-learn "
        "pandas joblib google-genai`, then set `GEMINI_API_KEY` in your environment."
    )
    st.code(
        "# PowerShell (current terminal)\n"
        "$env:GEMINI_API_KEY = \"YOUR_API_KEY\"\n"
        "streamlit run app.py\n\n"
        "# macOS/Linux (current terminal)\n"
        "export GEMINI_API_KEY=\"YOUR_API_KEY\"\n"
        "streamlit run app.py",
        language="shell",
    )
