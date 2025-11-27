import streamlit as st
import pandas as pd

from core.model_loader import load_models
from core.preprocessor import preprocess_input
from core.predictor import predict_row
from core.utils import compute_metrics, plot_actual_vs_pred


# -----------------------------------
# App Config
# -----------------------------------
st.set_page_config(
    page_title="UAE Retail Demand Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 UAE Retail Demand Forecasting Dashboard")
st.write("A simple, clear tool to help UAE retail teams forecast daily sales and detect demand spikes.")


# -----------------------------------
# Load Models
# -----------------------------------
with st.spinner("🔄 Loading forecasting models..."):
    clf_spike, model_normal, model_spike, features, le_prod, le_store = load_models()
st.success("Models loaded successfully ✔")


# -----------------------------------
# Tabs
# -----------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Upload & Predict",
    "📊 Visual Analysis",
    "🧾 Simple Explanation",
    "ℹ️ Model Info",
    "📘 About Project"
])


# ======================================================
# 📤 TAB 1 — Upload & Predict
# ======================================================
with tab1:
    st.subheader("Upload Your Sales Data")

    with st.expander("📘 What file should I upload?", expanded=True):
        st.markdown("""
A simple **CSV file** with:

- **Date**
- **Product ID**
- **Store ID**
- **Price**
- **Discount**
- **Inventory Level**
- **Competitor Pricing**
- *(Optional)* Units Sold → enables accuracy score
        """)

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:
        df_raw = pd.read_csv(file)
        st.write("Preview:")
        st.dataframe(df_raw.head())

        # Preprocess
        with st.spinner("Preparing data..."):
            X = preprocess_input(df_raw.copy(), le_prod, le_store, features)

        st.success("Data ready ✔")

        # Predict
        with st.spinner("Generating predictions..."):
            results = [predict_row(clf_spike, model_normal, model_spike, X.iloc[i:i+1])
                       for i in range(len(X))]

        df = df_raw.copy()
        df["Spike Probability"] = [r["spike_prob"] for r in results]
        df["Normal Prediction"] = [r["normal_pred"] for r in results]
        df["Spike Prediction"] = [r["spike_pred"] for r in results]
        df["Final Forecast"] = [r["blend_pred"] for r in results]

        st.subheader("📄 Predictions")
        st.dataframe(df.head())

        st.session_state["pred_df"] = df


# ======================================================
# 📊 TAB 2 — Visual Analysis
# ======================================================
with tab2:
    if "pred_df" not in st.session_state:
        st.info("Upload a file first.")
    else:
        df = st.session_state["pred_df"]

        if "Units Sold" in df.columns:
            st.subheader("📈 Accuracy")
            mae, rmse = compute_metrics(df["Units Sold"], df["Final Forecast"])

            col1, col2 = st.columns(2)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")

            st.subheader("📊 Actual vs Predicted")
            fig = plot_actual_vs_pred(df.rename(columns={"Final Forecast": "Pred_Blend"}))
            st.pyplot(fig)
        else:
            st.warning("Units Sold missing → accuracy not available.")


# ======================================================
# 🧾 TAB 3 — Simple Explanation (for non-technical users)
# ======================================================
with tab3:
    st.subheader("🧾 Easy Explanation")

    if "pred_df" not in st.session_state:
        st.info("Upload a file first.")
    else:
        df = st.session_state["pred_df"]

        st.markdown("""
### 🔍 What the numbers mean

- **Final Forecast** → How many units you're likely to sell  
- **Spike Probability** → Chance of a big sales jump  
- **Normal Prediction** → Sales on a regular day  
- **Spike Prediction** → Sales if the day becomes busy  
""")

        st.markdown("### 📋 Daily Breakdown")

        for _, row in df.iterrows():
            prob = row["Spike Probability"]
            final = row["Final Forecast"]

            if prob >= 0.7:
                status = "🔴 High Spike Risk — Increase stock"
            elif prob >= 0.4:
                status = "🟠 Medium Spike Chance — Monitor inventory"
            else:
                status = "🟢 Normal Day — Standard operations"

            st.markdown(f"""
---
**📅 Date:** {row.get('Date','N/A')}  
**🛒 Product:** {row.get('Product ID','N/A')}  
**🏬 Store:** {row.get('Store ID','N/A')}  

**📈 Forecast:** `{final:.1f}` units  
**🔥 Spike Chance:** `{prob*100:.1f}%`  
**📌 Status:** {status}
""")


# ======================================================
# ℹ️ TAB 4 — Model Info (Short + Recruiter Friendly)
# ======================================================
with tab4:
    st.subheader("ℹ️ Model Information")
    st.markdown(f"""
This forecasting system uses:

- **Spike Detector (XGBoost Classifier)**
- **Normal-Day Forecaster (XGBoost Regressor)**
- **Spike-Day Forecaster (XGBoost Regressor)**
- **Blended Model** combining both predictions

### Why this works well:
- Captures UAE weekend pattern (**Fri–Sat**)
- Handles Ramadan & Eid spikes
- Responds to promotions & discounts
- Uses recent demand patterns (lags, rolling averages)
- Adapts to price changes & competitor pricing

### Features used: {len(features)}
""")



# ======================================================
# 📘 TAB 5 — About Project (Simple + Recruiter-Focused)
# ======================================================
with tab5:
    st.subheader("📘 About This Project")
    st.markdown("""
This project predicts **daily retail demand** for UAE stores and highlights **spike days**  
(where demand suddenly increases).

### What the system solves
- Stockout prevention  
- Better inventory planning  
- Promotion planning  
- Weekend & event demand fluctuations  
- Product-level sales forecasting  

### Why UAE?
UAE retail is unique because of:
- Friday–Saturday weekends  
- Ramadan & Eid demand patterns  
- Tourism-driven fluctuations  
- Heavy discount seasons  
- National Day spikes  

### How the model was trained
- Trained in **Google Colab**  
- Feature engineering (lags, rolling averages, volatility, discount effects)  
- Three-model architecture (classifier + two regressors)  
- Exported as **JSON models** for full cross-platform support  

### Who is this for?
- Store managers  
- Retail operations teams  
- Planning & replenishment teams  
- Business analysts  

### One-line summary
**A simple, clear tool to predict daily product demand and detect spike days in UAE retail.**
""")
