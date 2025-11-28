import pickle
import json
import os

# Load XGBoost booster JSON helper
def load_xgb_json(path):
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def load_models():
    """
    Loads:
    - XGBoost classifier (spike detector)
    - XGBoost regressors (normal + spike)
    - Feature list
    - Label encoders (product + store)
    """
    # ---- Load classifier ----
    with open("models/clf_spike.pkl", "rb") as f:
        clf_spike = pickle.load(f)         # Do NOT set classes_ manually

    # ---- Load XGBoost regressors ----
    with open("models/model_normal.pkl", "rb") as f:
        model_normal = pickle.load(f)

    with open("models/model_spike.pkl", "rb") as f:
        model_spike = pickle.load(f)

    # ---- Load features ----
    with open("models/features.pkl", "rb") as f:
        features = pickle.load(f)

    # ---- Load encoders ----
    with open("models/le_prod.pkl", "rb") as f:
        le_prod = pickle.load(f)

    with open("models/le_store.pkl", "rb") as f:
        le_store = pickle.load(f)

    return clf_spike, model_normal, model_spike, features, le_prod, le_store
