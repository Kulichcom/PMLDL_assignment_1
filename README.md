# PMLDL Assignment 1: Loan Approval Prediction Pipeline

An automated MLOps pipeline that cleans data, trains a loan approval classifier, and deploys it as an API + web app, all running in Docker containers and re-triggered automatically every 5 minutes.

## What it does

Given an applicant's details (income, credit history, education, etc.), the model predicts whether their loan would be **Approved** or **Rejected**, based on patterns learned from the [Loan Prediction Problem dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).

## Pipeline stages

1. **Data engineering** (`code/datasets/prepare_data.py`) - loads the raw dataset, handles missing values, removes outliers, splits into train/test sets.
2. **Model engineering** (`code/models/train_model.py`) - encodes categorical features, trains a Logistic Regression classifier, evaluates it (accuracy, precision, recall, F1), logs everything to MLflow, and saves the trained model.
3. **Deployment** (`code/deployment/`) - a FastAPI service serves predictions from the trained model; a Streamlit app provides a form for entering applicant data and displaying the prediction. Each runs in its own Docker container, connected via Docker Compose.

## Repository structure

~~~
code/
  datasets/          data cleaning and splitting script
  models/            feature engineering, training, evaluation script
  deployment/
    api/             FastAPI service + Dockerfile
    app/             Streamlit app + Dockerfile
    docker-compose.yml
data/
  raw/               original dataset (train.csv, test.csv)
  processed/         cleaned, split data
models/              trained model + label encoders (model.joblib, encoders.joblib)
pipeline.py          orchestrator: runs all 3 stages in sequence
run_scheduler.sh     loops pipeline.py every 5 minutes
requirements.txt
README.md
~~~

## Setup

1. Clone the repo and create a virtual environment:
   ~~~
   git clone https://github.com/Kulichcom/PMLDL_assignment_1.git
   cd PMLDL_assignment_1
   python3 -m venv venv
   source venv/bin/activate
   ~~~
2. Install dependencies:
   ~~~
   pip install -r requirements.txt
   ~~~
3. Make sure Docker Desktop is installed and running.

## Running the pipeline manually (one-off)

~~~
python pipeline.py
~~~

This runs data cleaning, model training, and rebuilds/restarts the Docker containers, in that order.

## Running the pipeline automatically (every 5 minutes)

~~~
chmod +x run_scheduler.sh
nohup ./run_scheduler.sh &
~~~

This starts a background loop that runs `pipeline.py`, waits 5 minutes, and repeats indefinitely. All output is logged to `pipeline.log`.

To stop it:
~~~
jobs
kill %1
~~~

**Note:** this uses a simple shell loop rather than `cron`, since `cron` requires macOS Full Disk Access permissions that we chose not to grant for this project. The loop achieves the same "runs every 5 minutes" requirement while running entirely under the user's own permissions.

## Accessing the API and app

Once the containers are running (via `pipeline.py` or manually with `docker-compose up --build` inside `code/deployment/`):

- **API**: http://localhost:8000/docs (interactive Swagger UI)
- **App**: http://localhost:8501

## Model performance

On the held-out test set:
- Accuracy: ~0.82
- Precision: ~0.81
- Recall: ~0.96
- F1 score: ~0.88

Logged automatically via MLflow on every training run.
