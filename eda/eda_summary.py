# eda_summary.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    input_file = "cleaned_dataset.csv"
    df = pd.read_csv(input_file)

    print("=== Descriptive Statistics ===")
    print(df.describe())

    import os
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(df["stress_score"], kde=True, bins=10, color='skyblue')
    plt.title("Distribution of Average Stress Score")
    plt.xlabel("Stress Score")
    plt.ylabel("Count")
    plt.savefig(f"{plot_dir}/stress_score_distribution.png")
    plt.close()

    # Plot 2: Boxplots for each stress situation
    stress_columns = [
        "What is your stress level in these given situations [You have to submit an assignment in less than a day]",
        "What is your stress level in these given situations [A week before exams]",
        "What is your stress level in these given situations [Asking for an extra ketchup packet at a restaurant]",
        "What is your stress level in these given situations [Meeting a new person ]",
        "What is your stress level in these given situations [Asking for help]",
        "What is your stress level in these given situations [Confronting someone]",
        "What is your stress level in these given situations [Doing something without help]"
    ]


    short_labels = [
        "submitting in less than a day",
        "week before exams",
        "asking for ketchup",
        "meeting a new person",
        "asking for help",
        "confronting someone",
        "doing something without help"
    ]

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[stress_columns])
    plt.xticks(ticks=range(len(short_labels)), labels=short_labels, rotation=25, ha='right')
    plt.title("Boxplots of Stress Levels Across Situations")
    plt.ylabel("Stress Level (1–5)")
    plt.ylim(0, 6)  # Add vertical spacing
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/stress_levels_boxplot.png")
    plt.show()

    # Plot 3: Correlation heatmap
    corr = df[stress_columns + ["stress_score"]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Stress Features")
    plt.savefig(f"{plot_dir}/stress_correlation_heatmap.png")
    plt.close()

    print("EDA complete. Plots saved in 'plots/' folder.")

if __name__ == "__main__":
    main()
