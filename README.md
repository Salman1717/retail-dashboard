# 📈 UAE Retail Demand Forecasting Dashboard
AI-Powered Daily Sales Prediction & Spike Detection for UAE Retail

## 🔗 Live Dashboard:
https://retail-dashboard-b5btlgjtemmsvq6yaxsekr.streamlit.app/

## 🔗 Model Training Notebook (Google Colab):
https://colab.research.google.com/drive/1RbQzZoNmiFp6hDyZUxHFH8yp1Td23Oj6?usp=sharing

## 🧠 Overview

This project builds a complete UAE-focused retail demand forecasting system, capable of predicting daily unit sales and identifying spike days (e.g., Eid, promotions, weekend rush).

UAE retail behavior is unique due to:

- 🇦🇪 Friday–Saturday weekends
- 🌙 Ramadan & Eid demand surges
- 🎉 Heavy promotions (Dubai Shopping Festival, National Day)
- 🌤️ Weather + tourism seasons

This system models all of these patterns through machine learning + engineered features.

## 🌟 Key Features

### 🚀 1. Upload → Predict → Download
- Upload CSV
- Get demand forecasts instantly
- Download predictions as CSV

### 📊 2. Visual Accuracy Analysis
- MAE
- RMSE
- Actual vs Predicted chart

### 🧠 3. Hybrid ML Architecture
- Spike Classifier (XGBoost)
- Normal-Day Forecaster
- Spike-Day Forecaster
- Blended Model for final accuracy

### ⚡ 4. Fallback Lightweight Model
For files with < 60 days of data, the system automatically uses a faster rules-based model.

### 👥 5. Non-Technical Summary
Human-readable insights for retail teams:
- 🔴 High Spike Risk — Increase stock

## 🧩 System Architecture
```
                ┌─────────────────────────┐
                │      Retail CSV         │
                │ (Date, Price, Stock...) │
                └─────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Preprocessing Layer   │
                 │ (Encoding, Features)    │
                 └─────────────┬──────────┘
                               │
                               ▼
          ┌────────────────────────────┐
          │     Spike Detector (CLS)   │
          │     XGBoost Classifier     │
          └──────────────┬────────────┘
                         │ spike_prob
                         ▼
┌────────────────────┐          ┌──────────────────────┐
│ Normal-Day Model   │          │ Spike-Day Model      │
│ XGBoost Regressor  │          │ XGBoost Regressor    │
└─────────┬──────────┘          └──────────┬───────────┘
          │                                 │
   normal_pred                         spike_pred
          └──────────────┬───────────────┘
                         ▼
              ┌─────────────────────┐
              │  Blended Prediction │
              └──────────┬──────────┘
                         ▼
                Final Forecast Output
```

## 📁 Dataset Format (CSV Input)

Your file must include:

| Column | Description |
|--------|-------------|
| Date | YYYY-MM-DD format |
| Product ID | e.g., P001 |
| Store ID | e.g., S001 |
| Price | Selling price |
| Discount | Discount applied |
| Inventory Level | Current stock |
| Competitor Pricing | Competitor price |
| Units Sold (optional) | Used to compute accuracy |

Recommended: At least 60 days of historical data per product-store.

## 📦 Project Structure
```
📂 retail-dashboard
│── app.py
│── requirements.txt
│── README.md
│
│── 📂 core
│     ├── model_loader.py
│     ├── preprocessor.py
│     ├── predictor.py
│     ├── utils.py
│
│── 📂 models
│     ├── clf_spike.json / pkl
│     ├── model_normal.json / pkl
│     ├── model_spike.json / pkl
│     ├── features.pkl
│     ├── le_prod.pkl
│     ├── le_store.pkl
│
│── 📂 data
│     ├── sample_data.csv
```

## ⚙️ Technologies Used

### 🧠 Machine Learning
- XGBoost Classifier (Spike Detection)
- XGBoost Regressors (Normal & Spike Forecasting)
- Time-series feature engineering
- Model export (JSON/Pickle)

### 🖥 Dashboard
- Streamlit
- Pandas
- Matplotlib

### 🔬 Training Platform
- Google Colab
- Scikit-learn
- XGBoost

## 🧠 Model Training Notebook

Full training pipeline including:
- Data Cleaning
- Label Encoding
- Feature Engineering
- Rolling Stats
- Lag Features
- Spike Classifier Training
- Regressor Training
- ROC Curve
- PR Curve
- Feature Importance
- Saving Models

🔗 Colab Link:
https://colab.research.google.com/drive/1RbQzZoNmiFp6hDyZUxHFH8yp1Td23Oj6?usp=sharing

## 🌍 UAE-Specific Behavior Modeled

This project is designed specifically for UAE retail patterns:
- Friday–Saturday weekend effect
- Ramadan night shopping surge
- Eid season hyper demand
- Tourism cycles (Nov–Mar peak)
- National Day promotions
- Weather-driven indoor/outdoor shopping changes
- Price competition sensitivity

This regional behavior dramatically improves forecasting accuracy.

## 🎯 Goal

Make retail forecasting simple, actionable, and UAE-specific — usable by store managers, not just data teams.

## 👤 Author

Salman Mhaskar
