# modeling.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def save_predictions(y_test, preds, model_name):
    df_preds = pd.DataFrame({
        "actual": y_test.values,
        "predicted": preds
    })
    df_preds.to_csv(f"{model_name.lower().replace(' ', '_')}_predictions.csv", index=False)

def main():
    # Load cleaned data
    df = pd.read_csv("cleaned_dataset.csv")

    # Clean target column
    col = "How many hours of actual sleep did you get on an average for the past month? (maybe different from the number of hours spent in bed)"
    df = df.rename(columns={col: "sleep_hours"})
    df["sleep_hours"] = df["sleep_hours"].astype(str).str.extract(r'(\d+\.?\d*)')[0]
    df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors='coerce')
    df.dropna(subset=["sleep_hours", "stress_score"], inplace=True)

    # Features and target
    feature_cols = [
        "What is your stress level in these given situations [You have to submit an assignment in less than a day]",
        "What is your stress level in these given situations [A week before exams]",
        "What is your stress level in these given situations [Asking for an extra ketchup packet at a restaurant]",
        "What is your stress level in these given situations [Meeting a new person ]",
        "What is your stress level in these given situations [Asking for help]",
        "What is your stress level in these given situations [Confronting someone]",
        "What is your stress level in these given situations [Doing something without help]",
        "stress_score"
    ]
    X = df[feature_cols]
    y = df["sleep_hours"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []

    # 1. Ridge
    ridge = Ridge().fit(X_train, y_train)
    ridge_preds = ridge.predict(X_test)
    save_predictions(y_test, ridge_preds, "Ridge")
    results.append({
        "Model": "Ridge",
        "RMSE": np.sqrt(mean_squared_error(y_test, ridge_preds)),
        "R2": r2_score(y_test, ridge_preds)
    })

    # 2. Lasso
    lasso = Lasso(alpha=0.1).fit(X_train, y_train)
    lasso_preds = lasso.predict(X_test)
    save_predictions(y_test, lasso_preds, "Lasso")
    results.append({
        "Model": "Lasso",
        "RMSE": np.sqrt(mean_squared_error(y_test, lasso_preds)),
        "R2": r2_score(y_test, lasso_preds)
    })

    # 3. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    save_predictions(y_test, rf_preds, "Random Forest")
    results.append({
        "Model": "Random Forest",
        "RMSE": np.sqrt(mean_squared_error(y_test, rf_preds)),
        "R2": r2_score(y_test, rf_preds)
    })

    # Save metrics to CSV
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("model_metrics.csv", index=False)

    print("✅ Predictions and metrics saved to CSV files.")

if __name__ == "__main__":
    main()
