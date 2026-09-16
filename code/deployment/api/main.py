from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Loan Approval API")

model = joblib.load("models/model.joblib")
encoders = joblib.load("models/encoders.joblib")

CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

class LoanApplication(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

@app.get("/")
def read_root():
    return {"message": "Loan Approval API is running"}

@app.post("/predict")
def predict(application: LoanApplication):
    data = application.dict()
    df = pd.DataFrame([data])

    for col in CATEGORICAL_COLS:
        df[col] = encoders[col].transform(df[col])

    prediction = model.predict(df)[0]
    label = encoders["Loan_Status"].inverse_transform([prediction])[0]

    result = "Approved" if label == "Y" else "Rejected"
    return {"prediction": result}
