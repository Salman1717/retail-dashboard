import pandas as pd
import numpy as np

# ---------------------------------------------------------
# Safe Label Encoding (handles unseen IDs)
# ---------------------------------------------------------
def safe_label_encode(encoder, values):
    known = set(encoder.classes_)
    output = []

    for v in values:
        if v in known:
            output.append(encoder.transform([v])[0])
        else:
            output.append(-1)  # unseen product or store
    return output


# ---------------------------------------------------------
# Compute time-series features for a single product-store group
# ---------------------------------------------------------
def compute_group_features(g):
    g = g.sort_values("Date").reset_index(drop=True)

    # Lags
    g["lag_1"] = g["Units Sold"].shift(1)
    g["lag_7"] = g["Units Sold"].shift(7)
    g["lag_14"] = g["Units Sold"].shift(14)

    # Rolling Features
    g["roll_mean_7"]  = g["Units Sold"].rolling(7, min_periods=1).mean()
    g["roll_mean_30"] = g["Units Sold"].rolling(30, min_periods=1).mean()
    g["roll_std_7"]   = g["Units Sold"].rolling(7, min_periods=1).std()
    g["roll_std_30"]  = g["Units Sold"].rolling(30, min_periods=1).std()

    # Expanding Mean
    g["expanding_mean"] = g["Units Sold"].expanding().mean()

    # Momentum Features
    g["diff_1"] = g["lag_1"] - g["lag_7"]
    g["diff_7"] = g["lag_7"] - g["lag_14"]

    return g


# ---------------------------------------------------------
# Main Preprocessing Pipeline
# ---------------------------------------------------------
def preprocess_input(df, le_prod, le_store, features):
    """
    Performs full preprocessing only when the dataset
    contains 60+ continuous days for each Product+Store group.
    Otherwise, Streamlit will switch to fallback model.
    """

    # -----------------------------------------------------
    # 1. Parse Date
    # -----------------------------------------------------
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        raise ValueError("❌ Missing required column: Date")

    # -----------------------------------------------------
    # 2. Encode Product & Store
    # -----------------------------------------------------
    df["product_id_enc"] = safe_label_encode(le_prod, df["Product ID"])
    df["store_id_enc"]   = safe_label_encode(le_store, df["Store ID"])

    # -----------------------------------------------------
    # 3. Calendar Features
    # -----------------------------------------------------
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["week"]        = df["Date"].dt.isocalendar().week.astype(int)
    df["month"]       = df["Date"].dt.month
    df["year"]        = df["Date"].dt.year
    df["is_weekend"]  = df["day_of_week"].isin([4,5]).astype(int)

    # -----------------------------------------------------
    # 4. Price Features
    # -----------------------------------------------------
    df["price_gap"] = df["Price"] - df["Competitor Pricing"]
    df["discount_ratio"] = df["Discount"] / (df["Price"] + 1e-9)

    # -----------------------------------------------------
    # 5. Time-Series Features (per product-store)
    # -----------------------------------------------------
    full_feature_df = []

    grouped = df.groupby(["Product ID", "Store ID"])

    for (p, s), g in grouped:

        # Need at least 60 days to compute meaningful lags
        if g["Date"].nunique() < 60:
            continue  

        g = compute_group_features(g)
        full_feature_df.append(g)

    if len(full_feature_df) == 0:
        # No valid group available
        return None  

    df = pd.concat(full_feature_df, ignore_index=True)

    # -----------------------------------------------------
    # 6. Fill Missing Columns
    # -----------------------------------------------------
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # -----------------------------------------------------
    # 7. Final feature ordering
    # -----------------------------------------------------
    df = df[features]

    return df
