# app.py (full cleaned version)
import streamlit as st
import pandas as pd
import numpy as np
import io
import os

from core.model_loader import load_models
from core.preprocessor import preprocess_input
from core.predictor import predict_row, fallback_predict_row
from core.utils import compute_metrics, plot_actual_vs_pred

# -----------------------------------
# Load Sample Data from File
# -----------------------------------
def load_sample_data():
    """Load sample data from external CSV file (data/testdata.csv)."""
    sample_path = "data/testdata.csv"
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    else:
        return None


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
# Helper: make merge-key columns (string-safe for joins)
# ======================================================
def make_merge_keys(df):
    # create string keys for reliable merging
    df2 = df.copy()
    if "Date" in df2.columns:
        # normalize date to ISO string
        df2["__date_key"] = pd.to_datetime(df2["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        df2["__date_key"] = ""
    df2["__prod_key"] = df2.get("Product ID", "").astype(str)
    df2["__store_key"] = df2.get("Store ID", "").astype(str)
    return df2[["__date_key", "__prod_key", "__store_key"]]


# ======================================================
# 📤 TAB 1 — Upload & Predict
# ======================================================
with tab1:
    st.subheader("Upload Your Sales Data")

    colA, colB = st.columns([2,1])
    file = colA.file_uploader("Upload CSV File", type=["csv"])
    sample_button = colB.button("📄 Try Sample Data (from data/testdata.csv)")

    if sample_button:
        sample_df = load_sample_data()
        if sample_df is None:
            st.error("Sample file not found at `data/testdata.csv`. Create the file or upload your CSV.")
        else:
            st.info(f"Loaded sample dataset with {len(sample_df)} records.")
            file = io.StringIO(sample_df.to_csv(index=False))

    if file:
        df_raw = pd.read_csv(file)

        # Basic validation
        if "Date" not in df_raw.columns or "Product ID" not in df_raw.columns or "Store ID" not in df_raw.columns:
            st.error("Your CSV must contain at least these columns: Date, Product ID, Store ID")
            st.stop()

        # Parse Date
        df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
        if df_raw["Date"].isna().any():
            st.warning("Some dates could not be parsed. Rows with invalid dates will be ignored for full-model grouping.")

        st.write("Preview:")
        st.dataframe(df_raw.head(10))

        # Summary KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("📅 Total Unique Dates", int(df_raw["Date"].nunique()))
        col2.metric("🛍️ Unique Products", int(df_raw["Product ID"].nunique()))
        col3.metric("🏬 Unique Stores", int(df_raw["Store ID"].nunique()))

        # ---------------------------
        # Determine which groups are eligible for full model
        # (Require at least 60 unique dates per product-store)
        # ---------------------------
        grouped = df_raw.groupby(["Product ID", "Store ID"])
        eligible_groups = set()
        for (prod, store), g in grouped:
            # count unique valid dates only
            valid_dates = g["Date"].dropna().dt.strftime("%Y-%m-%d").nunique()
            if valid_dates >= 60:
                eligible_groups.add((prod, store))

        st.write(f"Groups with >=60 days: {len(eligible_groups)}")

        # If none eligible, alert user and use fallback for all rows
        if len(eligible_groups) == 0:
            st.warning("No product-store group has 60+ days of history. Using fallback model for all rows.")
            use_full_model = False
        else:
            use_full_model = True

        # ---------------------------
        # Split rows into two sets:
        # - rows_for_full: rows belonging to eligible groups (if any)
        # - rows_for_fallback: remaining rows
        # ---------------------------
        # Build boolean mask
        df_raw["_group_tuple"] = list(zip(df_raw["Product ID"].astype(str), df_raw["Store ID"].astype(str)))
        if use_full_model:
            mask_full = df_raw["_group_tuple"].apply(lambda t: t in eligible_groups)
            df_full_rows = df_raw[mask_full].copy().reset_index(drop=True)
            df_fallback_rows = df_raw[~mask_full].copy().reset_index(drop=True)
        else:
            df_full_rows = pd.DataFrame(columns=df_raw.columns)
            df_fallback_rows = df_raw.copy().reset_index(drop=True)

        # ---------------------------
        # Predictions container
        # ---------------------------
        pred_rows = []

        # ---------------------------
        # 1) Full model predictions (if any)
        # ---------------------------
        if len(df_full_rows) > 0:
            st.info(f"Running full model for {len(df_full_rows)} rows (eligible groups).")

            # Preprocess only the filtered rows (preprocess_input expects history per group)
            # Note: preprocess_input was implemented to compute group features and return features in same relative order.
            X = preprocess_input(df_full_rows.copy(), le_prod, le_store, features)

            if X is None or len(X) == 0:
                st.warning("Preprocessing did not return usable feature rows for full model — falling back to fallback model for these rows.")
                # treat all those rows as fallback
                df_fallback_rows = pd.concat([df_fallback_rows, df_full_rows], ignore_index=True)
                df_full_rows = pd.DataFrame(columns=df_full_rows.columns)
            else:
                # Run full-model predictions on X (X rows correspond to df_full_rows rows)
                results = []
                for i in range(len(X)):
                    res = predict_row(clf_spike, model_normal, model_spike, X.iloc[i:i+1])
                    results.append(res)

                # Build pred_df for merging: include keys from df_full_rows for safe merge
                keys = make_merge_keys(df_full_rows).reset_index(drop=True)
                pred_df_full = pd.DataFrame(results)
                pred_df_full = pd.concat([keys, pred_df_full], axis=1)

                # Merge full-model predictions back to df_raw via keys
                # Convert keys to columns on original df
                df_full_rows_keys = make_merge_keys(df_full_rows).reset_index(drop=True)
                df_full_rows_out = pd.concat([df_full_rows.reset_index(drop=True), pred_df_full.drop(columns=["__date_key","__prod_key","__store_key"], errors="ignore")], axis=1)
                # We'll append these to pred_rows later
                pred_rows.append(df_full_rows_out)

        # ---------------------------
        # 2) Fallback predictions for remaining rows
        # ---------------------------
        if len(df_fallback_rows) > 0:
            st.info(f"Running fallback model for {len(df_fallback_rows)} rows (short history or ineligible groups).")
            fb_results = []
            for _, row in df_fallback_rows.iterrows():
                r = fallback_predict_row(row)
                fb_results.append(r)

            fb_pred_df = pd.DataFrame(fb_results)
            fb_keys = make_merge_keys(df_fallback_rows).reset_index(drop=True)
            fb_out = pd.concat([df_fallback_rows.reset_index(drop=True), fb_pred_df], axis=1)
            pred_rows.append(fb_out)

        # ---------------------------
        # 3) Combine results and present
        # ---------------------------
        if len(pred_rows) == 0:
            st.error("No predictions were produced. Check your data and try again.")
            st.stop()

        df_preds_combined = pd.concat(pred_rows, ignore_index=True, sort=False)

        # Some columns have different names (full model uses normal_pred/spike_pred/blend_pred, fallback uses forecast)
        # Normalize to common output fields: Spike Probability & Final Forecast
        if "blend_pred" in df_preds_combined.columns:
            df_preds_combined["Final Forecast"] = df_preds_combined["blend_pred"]
        if "forecast" in df_preds_combined.columns:
            # fallback column
            df_preds_combined["Final Forecast"] = df_preds_combined["forecast"]

        if "spike_prob" in df_preds_combined.columns:
            df_preds_combined["Spike Probability"] = df_preds_combined["spike_prob"]

        # Reattach these predictions to the original df_raw by keys (date/product/store)
        # Make keys on both frames and merge
        left_keys = make_merge_keys(df_raw).reset_index(drop=True)
        right_keys = make_merge_keys(df_preds_combined).reset_index(drop=True)

        # combine df_raw with predictions by concatenating key columns then merging on keys to preserve duplicates
        df_raw_keys = pd.concat([df_raw.reset_index(drop=True), left_keys], axis=1)
        df_preds_keys = pd.concat([df_preds_combined.reset_index(drop=True), right_keys], axis=1)

        # Perform a left merge that matches each original row to its prediction (if multiple matches, we keep the first)
        merged = pd.merge(
            df_raw_keys,
            df_preds_keys[["__date_key", "__prod_key", "__store_key", "Spike Probability", "Final Forecast"]].drop_duplicates(),
            how="left",
            left_on=["__date_key", "__prod_key", "__store_key"],
            right_on=["__date_key", "__prod_key", "__store_key"]
        )

        # If any rows still have no prediction (NaN) — apply fallback to those rows now
        rows_missing = merged["Final Forecast"].isna()
        if rows_missing.any():
            st.warning(f"{rows_missing.sum()} rows did not get a prediction from the full model merge — applying fallback to these rows.")
            fb_additional = []
            for _, row in merged[rows_missing].iterrows():
                fb_additional.append(fallback_predict_row(row))
            fb_add_df = pd.DataFrame(fb_additional)
            merged.loc[rows_missing, "Spike Probability"] = fb_add_df["spike_prob"].values
            merged.loc[rows_missing, "Final Forecast"] = fb_add_df["forecast"].values

        # Clean helper columns and show final DF
        merged = merged.drop(columns=[c for c in merged.columns if c.startswith("__")] + ["_group_tuple"], errors="ignore")

        st.subheader("📄 Predictions")
        st.dataframe(merged)

        # store for other tabs
        st.session_state["pred_df"] = merged

        # allow CSV download
        csv_data = merged.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Predictions CSV", data=csv_data, file_name="predictions.csv")


# ======================================================
# 📊 TAB 2 — Visual Analysis
# ======================================================
with tab2:
    if "pred_df" not in st.session_state:
        st.info("Upload a file first.")
    else:
        df = st.session_state["pred_df"]

        if "Units Sold" in df.columns and "Final Forecast" in df.columns:
            st.subheader("📈 Accuracy Metrics")
            mae, rmse = compute_metrics(df["Units Sold"], df["Final Forecast"])

            col1, col2 = st.columns(2)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("RMSE", f"{rmse:.2f}")

            st.subheader("📊 Actual vs Predicted")
            fig = plot_actual_vs_pred(df.rename(columns={"Final Forecast": "Pred_Blend"}))
            st.pyplot(fig)
        else:
            st.warning("Units Sold or Final Forecast missing → cannot compute accuracy.")


# ======================================================
# 🧾 TAB 3 — Simple Explanation
# ======================================================
with tab3:
    st.subheader("🧾 Simple Breakdown")

    if "pred_df" not in st.session_state:
        st.info("Upload a file first.")
    else:
        df = st.session_state["pred_df"]

        st.markdown("""
### 📌 What The Forecast Means
- **Final Forecast** → Expected units sold  
- **Spike Probability** → Chance of high-demand day  
        """)

        # Allow user to filter by high spike probability
        threshold = st.slider("Show only days with spike probability above:", 0.0, 1.0, 0.3, 0.05)

        filtered_df = df[df["Spike Probability"] >= threshold].copy()
        st.write(f"Showing {len(filtered_df)} rows with spike probability ≥ {threshold:.0%}")

        for _, row in filtered_df.iterrows():
            prob = row["Spike Probability"]
            final = row["Final Forecast"]

            if prob >= 0.7:
                status = "🔴 High Spike Risk — Increase stock"
            elif prob >= 0.4:
                status = "🟠 Medium Spike Chance — Monitor inventory"
            else:
                status = "🟢 Normal Day Expected"

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
# ℹ️ TAB 4 — Model Info
# ======================================================
with tab4:
    st.subheader("ℹ️ Model Details")
    st.markdown(f"""
This forecasting system uses:

- **XGBoost Spike Detector (Classifier)**
- **XGBoost Normal-Day Forecaster**
- **XGBoost Spike-Day Forecaster**
- **Weighted Blended Prediction**

Fallback model is used for datasets **smaller than 60 days** per product-store group.

**Total Features Used:** {len(features)}
""")


# ======================================================
# 📘 TAB 5 — About Project
# ======================================================
with tab5:
    st.subheader("📘 About This Project")
    st.markdown("""
This tool predicts **daily sales** and identifies **spike days** for UAE retail stores.

### 💡 What it helps with
- Avoid stockouts  
- Plan promotions  
- Adjust pricing  
- Forecast weekend & holiday demand  

### 🇦🇪 UAE Retail Patterns Considered
- Friday–Saturday weekends  
- Ramadan & Eid peaks  
- Tourism seasons  
- National Day promotions  
- Competitive pricing effects  

### 📁 Data Requirements
Upload a CSV with these columns:
- `Date` (YYYY-MM-DD format)
- `Product ID`
- `Store ID`
- `Price`
- `Discount`
- `Inventory Level`
- `Competitor Pricing`
- `Units Sold` (optional for predictions)

For best results, provide **60+ days** of historical data per product-store.
""")
