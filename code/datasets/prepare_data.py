import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/train.csv"
PROCESSED_DIR = "data/processed"

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop(columns=["Loan_ID"])

    # Fill missing categorical values with the most frequent value
    categorical_cols = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Fill missing numeric value with median
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])

    # Remove extreme outliers in income using the IQR method
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df

def split_and_save(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Loan_Status"])
    train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)
    print(f"Saved {len(train_df)} training rows and {len(test_df)} testing rows.")

if __name__ == "__main__":
    df = load_data(RAW_PATH)
    df = clean_data(df)
    split_and_save(df)
