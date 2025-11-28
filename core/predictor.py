import numpy as np
import pandas as pd
from datetime import datetime


# --------------------------------------------------------
# Main prediction function (for full models with 60+ days)
# --------------------------------------------------------
def predict_row(clf_spike, model_normal, model_spike, X_row):
    """
    Predict for a single row using the full trained models.
    
    Args:
        clf_spike: XGBoost classifier for spike detection
        model_normal: XGBoost regressor for normal days
        model_spike: XGBoost regressor for spike days
        X_row: DataFrame with one row of features
    
    Returns:
        dict with spike_prob, normal_pred, spike_pred, blend_pred
    """
    # Get spike probability
    spike_prob = clf_spike.predict_proba(X_row)[0, 1]
    
    # Get predictions from both models
    normal_pred = model_normal.predict(X_row)[0]
    spike_pred = model_spike.predict(X_row)[0]
    
    # Blend based on spike probability
    blend_pred = (1 - spike_prob) * normal_pred + spike_prob * spike_pred
    
    return {
        "spike_prob": spike_prob,
        "normal_pred": normal_pred,
        "spike_pred": spike_pred,
        "blend_pred": blend_pred
    }


# --------------------------------------------------------
# Fallback prediction (for small datasets < 60 days)
# --------------------------------------------------------
def fallback_predict_row(row):
    """
    Simple rule-based prediction for small datasets where time-series
    models cannot be properly trained.
    
    Args:
        row: pandas Series with raw data columns
    
    Returns:
        dict with spike_prob and forecast
    """
    # Base prediction starts with historical average or default
    base_units = 100  # default baseline
    
    # Extract features safely
    price = row.get("Price", 12.0)
    discount = row.get("Discount", 0.0)
    inventory = row.get("Inventory Level", 100)
    competitor_price = row.get("Competitor Pricing", 13.0)
    
    # Parse date for calendar features
    try:
        date = pd.to_datetime(row.get("Date"))
        day_of_week = date.dayofweek
        month = date.month
    except:
        day_of_week = 0
        month = 1
    
    # --- Rule-based adjustments ---
    
    # 1. Discount effect (higher discount = more sales)
    discount_multiplier = 1.0 + (discount / price) * 0.5 if price > 0 else 1.0
    
    # 2. Price competitiveness (lower price vs competitor = more sales)
    if competitor_price > 0:
        price_gap = competitor_price - price
        price_multiplier = 1.0 + (price_gap / competitor_price) * 0.3
    else:
        price_multiplier = 1.0
    
    # 3. Inventory constraint (low stock = fewer sales)
    inventory_multiplier = min(1.0, inventory / 150)
    
    # 4. Weekend boost (Friday=4, Saturday=5 in UAE)
    weekend_multiplier = 1.3 if day_of_week in [4, 5] else 1.0
    
    # 5. Holiday/seasonal patterns (simplified)
    # UAE National Day (Dec 2-3), Ramadan/Eid periods
    seasonal_multiplier = 1.0
    if month == 12:  # December (National Day)
        seasonal_multiplier = 1.4
    elif month in [4, 6, 7]:  # Common Eid months (approximate)
        seasonal_multiplier = 1.5
    
    # Calculate forecast
    forecast = (
        base_units 
        * discount_multiplier 
        * price_multiplier 
        * inventory_multiplier 
        * weekend_multiplier 
        * seasonal_multiplier
    )
    
    # Ensure reasonable bounds
    forecast = max(10, min(forecast, inventory * 0.95))
    
    # --- Spike probability estimation ---
    spike_indicators = 0
    
    # High discount
    if discount / price > 0.2:
        spike_indicators += 1
    
    # Weekend
    if day_of_week in [4, 5]:
        spike_indicators += 1
    
    # Good price vs competitor
    if price < competitor_price * 0.9:
        spike_indicators += 1
    
    # Holiday season
    if month in [4, 6, 7, 12]:
        spike_indicators += 1
    
    # High inventory (ready for spike)
    if inventory > 150:
        spike_indicators += 0.5
    
    # Convert to probability (0 to 1 scale)
    spike_prob = min(0.95, spike_indicators / 5.0)
    
    return {
        "spike_prob": spike_prob,
        "forecast": round(forecast, 1)
    }


# --------------------------------------------------------
# Batch prediction helper
# --------------------------------------------------------
def batch_predict(clf_spike, model_normal, model_spike, X_df):
    """
    Predict for multiple rows at once.
    
    Args:
        clf_spike: XGBoost classifier
        model_normal: XGBoost regressor for normal days
        model_spike: XGBoost regressor for spike days
        X_df: DataFrame with multiple rows of features
    
    Returns:
        DataFrame with predictions
    """
    results = []
    
    for i in range(len(X_df)):
        result = predict_row(clf_spike, model_normal, model_spike, X_df.iloc[i:i+1])
        results.append(result)
    
    return pd.DataFrame(results)


# --------------------------------------------------------
# Fallback batch prediction
# --------------------------------------------------------
def batch_fallback_predict(df_raw):
    """
    Apply fallback predictions to entire DataFrame.
    
    Args:
        df_raw: DataFrame with raw input data
    
    Returns:
        DataFrame with predictions
    """
    results = []
    
    for _, row in df_raw.iterrows():
        result = fallback_predict_row(row)
        results.append(result)
    
    return pd.DataFrame(results)