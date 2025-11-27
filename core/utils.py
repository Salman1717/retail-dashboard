import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse


def plot_actual_vs_pred(df):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df["Units Sold"].values, label="Actual Sales", linewidth=2)
    ax.plot(df["Pred_Blend"].values, label="Predicted (Blend)", linewidth=2)
    ax.legend()
    ax.set_title("Actual vs Predicted Demand")
    return fig
