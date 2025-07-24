# visualize_models.py

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_predictions(df, model_name):
    plt.figure(figsize=(6, 5))
    plt.scatter(df["actual"], df["predicted"], alpha=0.7)
    plt.plot([df["actual"].min(), df["actual"].max()],
             [df["actual"].min(), df["actual"].max()], 'r--')
    plt.xlabel("Actual Sleep Hours")
    plt.ylabel("Predicted Sleep Hours")
    plt.title(f"{model_name} - Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(f"plots/{model_name.lower().replace(' ', '_')}_pred_vs_actual.png")
    plt.close()

def plot_residuals(df, model_name):
    residuals = df["actual"] - df["predicted"]
    plt.figure(figsize=(6, 5))
    plt.scatter(df["predicted"], residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Sleep Hours")
    plt.ylabel("Residuals")
    plt.title(f"{model_name} - Residuals")
    plt.tight_layout()
    plt.savefig(f"plots/{model_name.lower().replace(' ', '_')}_residuals.png")
    plt.close()

def plot_metrics_bar(metrics_df):
    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["Model"], metrics_df["RMSE"], color='skyblue', label='RMSE')
    plt.ylabel("RMSE")
    plt.title("Model Comparison - RMSE")
    plt.tight_layout()
    plt.savefig("plots/model_rmse_comparison.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(metrics_df["Model"], metrics_df["R2"], color='salmon', label='R²')
    plt.ylabel("R²")
    plt.title("Model Comparison - R²")
    plt.tight_layout()
    plt.savefig("plots/model_r2_comparison.png")
    plt.close()

def main():
    os.makedirs("plots", exist_ok=True)

    models = ["Ridge", "Lasso", "Random Forest"]

    for model in models:
        file_path = f"{model.lower().replace(' ', '_')}_predictions.csv"
        df = pd.read_csv(file_path)
        plot_predictions(df, model)
        plot_residuals(df, model)

    metrics_df = pd.read_csv("model_metrics.csv")
    plot_metrics_bar(metrics_df)

    print("✅ All plots saved in /plots folder.")

if __name__ == "__main__":
    main()
