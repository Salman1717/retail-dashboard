def predict_row(clf_spike, model_normal, model_spike, row_df):
    prob = clf_spike.predict_proba(row_df)[0, 1]
    pred_normal = model_normal.predict(row_df)[0]
    pred_spike = model_spike.predict(row_df)[0]
    blended = (1 - prob) * pred_normal + prob * pred_spike
    
    return {
        "spike_prob": prob,
        "normal_pred": pred_normal,
        "spike_pred": pred_spike,
        "blend_pred": blended
    }
