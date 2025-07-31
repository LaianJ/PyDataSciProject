# eda.py

import pandas as pd

def main():
    input_path = "Dataset.xlsx"  
    output_path = "cleaned_dataset.csv"

    df = pd.read_excel(input_path, sheet_name='Form Responses 1')

    stress_columns = [
        "What is your stress level in these given situations [You have to submit an assignment in less than a day]",
        "What is your stress level in these given situations [A week before exams]",
        "What is your stress level in these given situations [Asking for an extra ketchup packet at a restaurant]",
        "What is your stress level in these given situations [Meeting a new person ]",
        "What is your stress level in these given situations [Asking for help]",
        "What is your stress level in these given situations [Confronting someone]",
        "What is your stress level in these given situations [Doing something without help]"
    ]

    stress_mapping = {
        "not stressed": 1,
        "mild": 2,
        "moderate": 3,
        "severe": 4,
        "very severe": 5
    }

    for col in stress_columns:
        df[col] = df[col].map(stress_mapping)

    df["stress_score"] = df[stress_columns].mean(axis=1)

    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")

if __name__ == "__main__":
    main()
