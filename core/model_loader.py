import xgboost as xgb
import pickle
import numpy as np

def load_models():
    # load classifier (sklearn wrapper)
    clf_spike = xgb.XGBClassifier()
    clf_spike.load_model("models/clf_spike.json")

    # When loading JSON, sklearn-wrapper metadata may be missing.
    # For a binary spike detector, set these manually:
    clf_spike.n_classes_ = 2
    clf_spike.classes_ = np.array([0, 1])

    # load regressors
    model_normal = xgb.XGBRegressor()
    model_normal.load_model("models/model_normal.json")

    model_spike = xgb.XGBRegressor()
    model_spike.load_model("models/model_spike.json")

    # load pickled objects
    features = pickle.load(open("models/features.pkl", "rb"))
    le_prod = pickle.load(open("models/le_prod.pkl", "rb"))
    le_store = pickle.load(open("models/le_store.pkl", "rb"))

    return clf_spike, model_normal, model_spike, features, le_prod, le_store
