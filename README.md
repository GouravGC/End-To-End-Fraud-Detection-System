# 💳 End-to-End Credit Card Fraud Detection System

A production-ready **Machine Learning** application for detecting fraudulent credit card transactions. The project follows a modular architecture, provides a user-friendly **Streamlit** interface for real-time predictions, and is designed for deployment and scalability.

---

# 🚀 Project Overview

This project transforms a complete Machine Learning workflow into a clean, production-ready Python application.

It includes:

* Modular project architecture
* Machine Learning training and prediction pipelines
* Interactive Streamlit web application
* Pre-trained model inference
* Structured logging and exception handling
* Deployment-ready codebase

The application predicts whether a credit card transaction is **Legitimate** or **Fraudulent** using a trained machine learning model.

---

# 📊 Dataset

This project is based on the popular **Credit Card Fraud Detection Dataset**.

The dataset contains:

* Time
* Transaction Amount
* 28 anonymized PCA features (V1–V28)
* Target variable (`Class`)

> **Note:** The dataset is **not included** in this repository because it exceeds GitHub's file size limits. Download it separately if you want to retrain the models.

---

# 🎯 Problem Statement

Credit card fraud is a highly imbalanced classification problem where fraudulent transactions represent only a very small percentage of all transactions.

The objective of this project is to accurately identify fraudulent transactions while minimizing false positives and preserving high recall.

---

# ✨ Key Features

* ✅ End-to-End Machine Learning Pipeline
* ✅ Production-Ready Modular Architecture
* ✅ Streamlit Web Application
* ✅ Real-Time Fraud Prediction
* ✅ Reusable Model Artifacts
* ✅ Structured Logging
* ✅ Exception Handling
* ✅ GitHub Ready
* ✅ Deployment Ready
* ✅ Docker Ready (Upcoming)

---

# 🏗️ Project Structure

```text
End-To-End-Fraud-Detection-System/
│
├── artifacts/
│   ├── models/
│   ├── feature_names.pkl
│   ├── scaler.pkl
│   └── other trained model artifacts
│
├── notebooks/
│
├── src/
│   ├── components/
│   ├── constants/
│   ├── pipeline/
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── app.py
├── main.py
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/GouravGC/End-To-End-Fraud-Detection-System.git
```

Move into the project directory:

```bash
cd End-To-End-Fraud-Detection-System
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application provides:

* Transaction feature input
* Real-time prediction
* Fraud probability
* Transaction summary

---

# 🤖 Machine Learning Models

The project includes multiple trained models for experimentation and comparison, including:

* Logistic Regression
* SMOTE Logistic Regression
* ADASYN Logistic Regression
* Weighted Logistic Regression
* Random Forest
* Weighted Random Forest
* Balanced Random Forest
* XGBoost

The deployed application uses the best-performing saved model for inference.

---

# 🛠️ Technologies Used

* Python
* Scikit-learn
* XGBoost
* Streamlit
* Pandas
* NumPy
* Joblib
* Matplotlib
* Seaborn
* Imbalanced-learn

---

# 📈 Results

The project evaluates multiple machine learning models using techniques designed for highly imbalanced datasets.

Evaluation includes:

* Precision
* Recall
* F1-Score
* Precision-Recall Curve
* Model Comparison

The deployed application performs inference using the saved production model without retraining.

---

# 🌐 Live Demo

**Streamlit App**

> Add your deployed Streamlit URL here.

---

# 📸 Application Preview

Add screenshots of:

* Home Page
* Prediction Result
* Fraud Detection Example

---

# 🔮 Future Improvements

* Docker Containerization
* CI/CD Pipeline
* Cloud Deployment
* MLflow Integration
* Model Monitoring
* Kubernetes Deployment

---

# 👨‍💻 Author

**Gourav Chhatwani**

GitHub: https://github.com/GouravGC

---
