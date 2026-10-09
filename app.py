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
        :root {
            --navy: #12304a;
            --navy-deep: #0b2033;
            --blue: #2463eb;
            --blue-soft: #eaf1ff;
            --ink: #172b3a;
            --muted: #657789;
            --line: #dfe7ef;
            --surface: #ffffff;
            --page: #f3f6fa;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 0%, #e4edff 0, transparent 28rem),
                var(--page);
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(243, 246, 250, 0.85);
        }

        .block-container {
            max-width: 1000px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        .hero {
            background: linear-gradient(125deg, #0b2033 0%, #17456b 62%, #2463a4 100%);
            color: #fff;
            padding: 2rem 2.1rem;
            border-radius: 22px;
            margin-bottom: 1.25rem;
            box-shadow: 0 16px 40px rgba(17, 48, 74, 0.17);
            position: relative;
            overflow: hidden;
        }

        .hero:after {
            content: "";
            position: absolute;
            width: 210px;
            height: 210px;
            border: 1px solid rgba(255,255,255,.16);
            border-radius: 50%;
            right: -55px;
            top: -85px;
            box-shadow: 0 0 0 24px rgba(255,255,255,.04),
                        0 0 0 48px rgba(255,255,255,.03);
        }

        .hero-kicker {
            color: #c9dcff;
            text-transform: uppercase;
            letter-spacing: .13em;
            font-size: .76rem;
            font-weight: 700;
            margin-bottom: .6rem;
        }

        .hero-title {
            font-size: clamp(1.8rem, 4vw, 2.55rem);
            line-height: 1.15;
            font-weight: 750;
            margin: 0 0 .65rem 0;
            color: #fff;
        }

        .hero-copy {
            max-width: 650px;
            color: #e1ebf7;
            font-size: 1rem;
            line-height: 1.6;
            margin: 0;
        }

        .trust-row {
            display: flex;
            flex-wrap: wrap;
            gap: .55rem;
            margin-top: 1.2rem;
        }

        .trust-pill {
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            border: 1px solid rgba(255,255,255,.22);
            background: rgba(255,255,255,.09);
            color: #f3f7ff;
            padding: .4rem .7rem;
            border-radius: 999px;
            font-size: .78rem;
        }

        .section-intro {
            margin: 1.25rem 0 .35rem 0;
            color: var(--navy);
            font-size: 1.12rem;
            font-weight: 750;
        }

        .section-subtitle {
            color: var(--muted);
            font-size: .9rem;
            margin: 0 0 .85rem 0;
        }

        div[data-testid="stForm"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.25rem 1.4rem 1.4rem 1.4rem;
            box-shadow: 0 8px 26px rgba(24, 50, 75, .055);
        }

        div[data-testid="stForm"] label {
            color: #31495d;
            font-weight: 600;
            font-size: .9rem;
        }

        div[data-testid="stNumberInput"] input,
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
            border-radius: 10px;
        }

        div[data-testid="stNumberInput"] input:focus,
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
            border-color: var(--blue);
            box-shadow: 0 0 0 1px var(--blue);
        }

        .stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(100deg, #17456b, #2463eb);
            color: #fff;
            font-weight: 700;
            font-size: 1rem;
            border: 0;
            border-radius: 11px;
            min-height: 3rem;
            box-shadow: 0 7px 16px rgba(36, 99, 235, .18);
            transition: transform .15s ease, box-shadow .15s ease;
        }

        .stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            color: #fff;
            border: 0;
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(36, 99, 235, .25);
        }

        .stAlert {
            border-radius: 14px;
        }

        .result-note {
            color: var(--muted);
            font-size: .85rem;
            margin-top: .75rem;
        }

        hr {
            border-color: var(--line);
        }

        @media (max-width: 640px) {
            .block-container {
                padding: 1rem .75rem 2rem .75rem;
            }
            .hero {
                padding: 1.45rem 1.25rem;
                border-radius: 17px;
            }
            div[data-testid="stForm"] {
                padding: 1rem .9rem;
                border-radius: 15px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Smart lending • Application portal</div>
        <div class="hero-title">🏦 Loan Application</div>
        <p class="hero-copy">
            Enter the applicant details below to receive a machine-learning
            prediction. Complete each section carefully for a consistent result.
        </p>
        <div class="trust-row">
            <span class="trust-pill">✦ AI-assisted prediction</span>
            <span class="trust-pill">✓ Guided application</span>
            <span class="trust-pill">⌁ Quick result</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "Educational prediction only — this is not a lender decision or a guarantee "
    "of loan approval."
)

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
    st.markdown('<div class="section-intro">01 · 👤 Personal details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Basic information about the applicant.</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-intro">02 · 💳 Financial details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Account and existing loan information.</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-intro">03 · 📞 Contact details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Recent campaign and contact history.</div>', unsafe_allow_html=True)
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


st.markdown(
    '<p class="result-note">Loan Application Portal · Predictions are generated by a trained machine-learning model and should be reviewed by a qualified decision-maker.</p>',
    unsafe_allow_html=True,
)
