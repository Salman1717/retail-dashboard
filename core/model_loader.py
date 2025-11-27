import xgboost as xgb
import pickle

def load_models():
    clf_spike = xgb.XGBClassifier()
    clf_spike.load_model("models/clf_spike.json")

    model_normal = xgb.XGBRegressor()
    model_normal.load_model("models/model_normal.json")

    model_spike = xgb.XGBRegressor()
    model_spike.load_model("models/model_spike.json")

    features = pickle.load(open("models/features.pkl", "rb"))
    le_prod = pickle.load(open("models/le_prod.pkl", "rb"))
    le_store = pickle.load(open("models/le_store.pkl", "rb"))

    return clf_spike, model_normal, model_spike, features, le_prod, le_store
