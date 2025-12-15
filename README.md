# Readmission-risk-prediction-api
End-to-end machine learning pipeline to predict hospital readmission risk, including data preprocessing, model training, and a FastAPI-based prediction service with a working REST API.
# 🏥 Hospital Readmission Prediction – End-to-End ML Pipeline

## 📌 Project Overview

This project implements a complete **end-to-end machine learning pipeline** to predict the probability of hospital readmission using patient-level clinical and administrative data.

The objective was not only to train a predictive model, but also to **serve it as a reliable production-style API**, handling real-world issues such as incomplete inputs, preprocessing consistency, and deployment challenges.

The final deliverable is a **working local FastAPI service** that exposes a `/predict` endpoint for inference.

---

## 🎯 Problem Statement

Hospital readmissions are expensive and often preventable.  
The goal of this project is to **predict the likelihood of patient readmission** based on demographic, clinical, and treatment-related features so that early interventions can be planned.

This is modeled as a **binary classification problem**, returning a probability score using `predict_proba`.

---

## 📊 Data Understanding & EDA

The dataset includes:
- Patient demographics (age, gender, race)
- Hospital utilization metrics (time in hospital, procedures)
- Diagnosis codes (`diag_1`, `diag_2`, `diag_3`)
- Medication and treatment information
- Admission and discharge details

### Key EDA Findings
- Numerical features required scaling and missing-value handling
- Categorical features had high cardinality and required encoding
- Diagnosis fields needed consistent preprocessing
- **Real-world inference requests often lack many training features**

These findings directly influenced the modeling and API design.

---

## ⚙️ Modeling Approach

### Pipeline Design

A **scikit-learn Pipeline** was used to prevent train–serve skew and ensure consistent preprocessing.

**Pipeline components:**
1. `ColumnTransformer`
   - Numerical features: imputation + scaling
   - Categorical features: imputation + encoding
2. Probabilistic classifier supporting `predict_proba`
3. End-to-end pipeline serialized with `joblib`

The trained model pipeline is stored as:

model.joblib


---

## 🚀 API Service (FastAPI)

A REST API was built using **FastAPI** to serve predictions.

### Endpoints

#### Health Check


GET /health


Response:
```json
{
  "status": "ok"
}
```

Prediction
POST /predict

Example request:

{
  "records": [
    {
      "data": {
        "time_in_hospital": 3,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 10,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "number_diagnoses": 5,
        "change": "No",
        "diabetesMed": "Yes"
      }
    }
  ]
}


Example response:

{
  "predictions": [0.56]
}


⚠️ Challenges Faced & How They Were Resolved
1. Missing Feature Errors (500 Internal Server Error)

Issue:
The model failed during inference due to missing columns.

Cause:
The training pipeline expected more features than those provided in inference requests.

Resolution:

Extracted expected numerical and categorical columns from the trained pipeline

Automatically filled missing numerical columns with 0

Filled missing categorical columns with "Unknown"

Ensured column order exactly matched training

This made the API robust to partial real-world inputs.

2. Pipeline Structure Mismatch

Issue:
KeyError: 'preprocessor' occurred due to incorrect assumptions about pipeline step names.

Resolution:

Inspected the actual trained pipeline

Safely accessed preprocessing components without hard-coded assumptions

3. Uvicorn / Port Binding Issues

Issue:
Local server failed to respond (connection refused).

Resolution:

Ensured Uvicorn was running successfully before testing

Verified /health locally prior to ngrok exposure

4. ngrok Connection Errors

Issues Encountered:

Multiple active ngrok sessions

ngrok started before the local service was live

Resolution:

Killed stale ngrok sessions

Started FastAPI first, verified locally, then exposed via ngrok


🏃 How to Run Locally
1. Install Dependencies
pip install fastapi uvicorn pandas scikit-learn joblib pyngrok

2. Start the API
python -m uvicorn app:app --host 0.0.0.0 --port 8000

3. Verify Health
curl http://localhost:8000/health

🌐 Optional: Expose via ngrok
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(public_url)


Swagger UI:

https://<ngrok-id>.ngrok-free.app/docs

✅ Final Outcome

✔ Fully functional local prediction service

✔ End-to-end ML pipeline with consistent preprocessing

✔ Robust handling of missing inference features

✔ Clean REST API with documented schema

✔ Production-style debugging and validation completed

📈 Future Improvements

Model explainability (SHAP)

Batch prediction endpoint

Monitoring and drift detection

Authentication and logging

Dockerization

📦 Deliverables Completed

✔ Modeling and service code

✔ Working local FastAPI service

✔ Example request/response

✔ Comprehensive documentation

✔ Ready for video walkthrough

