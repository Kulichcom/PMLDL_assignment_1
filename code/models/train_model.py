import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/model.joblib"
ENCODERS_PATH = "models/encoders.joblib"

CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
TARGET_COL = "Loan_Status"

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df

def engineer_features(train_df, test_df):
    encoders = {}
    for col in CATEGORICAL_COLS + [TARGET_COL]:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        encoders[col] = le

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    return X_train, y_train, X_test, y_test, encoders

def train_and_evaluate(X_train, y_train, X_test, y_test):
    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            print(f"{name}: {value:.4f}")

        mlflow.sklearn.log_model(model, "model")
        return model

def save_artifacts(model, encoders):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Encoders saved to {ENCODERS_PATH}")

if __name__ == "__main__":
    train_df, test_df = load_data()
    X_train, y_train, X_test, y_test, encoders = engineer_features(train_df, test_df)
    model = train_and_evaluate(X_train, y_train, X_test, y_test)
    save_artifacts(model, encoders)
