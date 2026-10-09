import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Application",
    page_icon="🏦",
    layout="centered",
)

MODEL_PATH = Path(__file__).resolve().parent / "model_prod_files.pkl"


@st.cache_resource
def load_model():
    """Load the fitted scikit-learn Pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Place model_prod_files.pkl in the same folder as app.py."
        )

    pipeline = joblib.load(MODEL_PATH)

    if not hasattr(pipeline, "predict") or not hasattr(pipeline, "named_steps"):
        raise TypeError(
            "model_prod_files.pkl does not contain the expected fitted "
            "scikit-learn Pipeline."
        )

    return pipeline


try:
    model = load_model()
except Exception as exc:
    st.error("Unable to load the trained model.")
    st.exception(exc)
    st.stop()


def get_encoder_categories(pipeline, column_name, fallback):
    """
    Return the categories learned by the fitted OneHotEncoder.
    This helps prevent inputs that the model was never trained to handle.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        cat_columns = None
        cat_encoder = None

        for transformer_name, transformer, columns in preprocessor.transformers_:
            if transformer_name == "cat":
                cat_columns = list(columns)
                cat_encoder = transformer
                break

        if cat_columns is not None and cat_encoder is not None:
            if column_name in cat_columns:
                index = cat_columns.index(column_name)
                values = list(cat_encoder.categories_[index])
                if values:
                    return [str(value) for value in values]
    except (KeyError, AttributeError, IndexError, TypeError):
        pass

    return fallback


def get_education_categories(pipeline):
    """Return education categories configured in the fitted OrdinalEncoder."""
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        for transformer_name, transformer, columns in preprocessor.transformers_:
            if transformer_name == "ord" and "education" in list(columns):
                return [str(value) for value in transformer.categories_[0]]
    except (AttributeError, IndexError, TypeError):
        pass

    return ["primary", "secondary", "tertiary"]


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f5f7fa;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 900px;
        }
        .stButton > button {
            background-color: #1a3c5e;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 0.6rem 2rem;
            width: 100%;
            border: none;
        }
        .stButton > button:hover {
            background-color: #25527e;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.title("🏦 Loan Application Form")
st.write(
    "Enter the details below to obtain a model-based prediction. "
    "Avoid entering unnecessary personal information."
)
st.info(
    "This application provides an educational machine-learning prediction. "
    "It is not an actual lender decision or a guarantee of loan approval."
)
st.divider()

# Build select-box options from the categories learned during training.
job_options = get_encoder_categories(
    model,
    "job",
    [
        "admin", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown",
    ],
)
marital_options = get_encoder_categories(
    model, "marital", ["single", "married", "divorced"]
)
contact_options = get_encoder_categories(
    model, "contact", ["cellular", "telephone", "unknown"]
)
month_options = get_encoder_categories(
    model,
    "month",
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"],
)
poutcome_options = get_encoder_categories(
    model, "poutcome", ["unknown", "failure", "success", "other"]
)
education_options = get_education_categories(model)

# -------------------------------------------------------------------
# Form
# -------------------------------------------------------------------
with st.form("loan_form"):
    st.subheader("👤 Personal Details")
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        marital = st.selectbox("Marital Status", marital_options)
        education = st.selectbox("Education Level", education_options)

    with c2:
        job = st.selectbox("Occupation", job_options)
        balance = st.number_input(
            "Account Balance (€)", value=1000, step=100
        )

    st.subheader("💳 Financial Details")
    c3, c4 = st.columns(2)

    with c3:
        default_val = st.selectbox(
            "Any existing credit default?", ["no", "yes"]
        )
        housing = st.selectbox("Do you have a housing loan?", ["no", "yes"])

    with c4:
        personal = st.selectbox("Do you have a personal loan?", ["no", "yes"])
        previous = st.number_input(
            "Number of previous contacts", min_value=0, value=0, step=1
        )

    st.subheader("📞 Contact Details")
    c5, c6 = st.columns(2)

    with c5:
        contact = st.selectbox("Preferred contact type", contact_options)
        month = st.selectbox("Month of last contact", month_options)

    with c6:
        day = st.number_input(
            "Day of last contact", min_value=1, max_value=31, value=15
        )
        duration = st.number_input(
            "Last call duration (seconds)", min_value=0, value=180, step=10
        )
        pdays = st.number_input(
            "Days since last contacted (-1 if never)",
            min_value=-1,
            value=-1,
            step=1,
        )
        poutcome = st.selectbox(
            "Outcome of previous contact", poutcome_options
        )

    submitted = st.form_submit_button(
        "Submit Application", use_container_width=True
    )

# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------
if submitted:
    campaign = 1  # Fixed internally, as in the original application.

    # Column names must match the original training dataset.
    input_df = pd.DataFrame(
        [{
            "age": int(age),
            "job": job,
            "marital": marital,
            "education": education,
            "default": 1 if default_val == "yes" else 0,
            "balance": float(balance),
            "housing_loan": 1 if housing == "yes" else 0,
            "personal_loan": 1 if personal == "yes" else 0,
            "contact": contact,
            "day": int(day),
            "month": month,
            "duration": int(duration),
            "campaign": campaign,
            "pdays": int(pdays),
            "previous": int(previous),
            "poutcome": poutcome,
        }]
    )

    # Validate that the form matches the feature schema used for training.
    expected_columns = list(getattr(model, "feature_names_in_", input_df.columns))
    missing_columns = [col for col in expected_columns if col not in input_df.columns]
    extra_columns = [col for col in input_df.columns if col not in expected_columns]

    if missing_columns or extra_columns:
        st.error("The form inputs do not match the model's training features.")
        if missing_columns:
            st.write("Missing columns:", missing_columns)
        if extra_columns:
            st.write("Unexpected columns:", extra_columns)
        st.stop()

    input_df = input_df[expected_columns]

    try:
        with st.spinner("🔍 Evaluating the application details..."):
            time.sleep(0.5)
            prediction = model.predict(input_df)[0]

        st.divider()

        # Preserve the same result messages as the original application.
        approved = int(prediction) == 1

        if approved:
            st.success("## ✅ Congratulations! Your loan has been **Approved**.")
            st.markdown(
                "Based on the details provided, your application has been "
                "accepted. Our team will contact you shortly."
            )
        else:
            st.error("## ❌ We're sorry. Your loan has been **Rejected**.")
            st.markdown(
                "Based on the details provided, we are unable to approve "
                "your application at this time."
            )

    except Exception as exc:
        st.error(
            "Prediction failed. Check that the model file was created from "
            "the same dataset schema and that all preprocessing dependencies "
            "are compatible with this environment."
        )
        st.exception(exc)
