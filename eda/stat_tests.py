# stat_tests.py

import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

def main():
    # Load cleaned dataset
    df = pd.read_csv("cleaned_dataset.csv")

    # Original column name from Excel
    original_col = "How many hours of actual sleep did you get on an average for the past month? (maybe different from the number of hours spent in bed)"
    
    # Rename column for easier use in formulas
    df = df.rename(columns={original_col: "sleep_hours"})

    # Extract numeric values from text like "6 hours", "about 5", etc.
    df["sleep_hours"] = df["sleep_hours"].astype(str).str.extract(r'(\d+\.?\d*)')[0]
    df["sleep_hours"] = pd.to_numeric(df["sleep_hours"], errors='coerce')

    # Drop rows with missing values
    df_clean = df.dropna(subset=["sleep_hours", "stress_score"])

    print(f"Number of valid rows after cleaning: {len(df_clean)}")

    if len(df_clean) < 2:
        print("ERROR: Not enough valid data to perform statistical tests.")
        return

    # Correlation Test
    print("=== Correlation Test ===")
    corr_pearson, pval_pearson = stats.pearsonr(df_clean["stress_score"], df_clean["sleep_hours"])
    print(f"Pearson correlation: {corr_pearson:.3f}, p-value: {pval_pearson:.4f}")

    # T-Test: Low vs. High Stress
    print("\n=== T-Test (Low vs. High Stress) ===")
    df_clean["stress_group"] = df_clean["stress_score"].apply(lambda x: "low" if x <= 2 else "high")
    low_group = df_clean[df_clean["stress_group"] == "low"]["sleep_hours"]
    high_group = df_clean[df_clean["stress_group"] == "high"]["sleep_hours"]
    t_stat, p_val = stats.ttest_ind(low_group, high_group, equal_var=False)
    print(f"T-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")

    # ANOVA Test
    print("\n=== ANOVA (Stress Score Binned) ===")
    df_clean["stress_bin"] = pd.cut(df_clean["stress_score"], bins=[0, 2, 3, 4, 5],
                                    labels=["Very Low", "Moderate", "High", "Very High"])
    model = smf.ols("sleep_hours ~ C(stress_bin)", data=df_clean).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

if __name__ == "__main__":
    main()
