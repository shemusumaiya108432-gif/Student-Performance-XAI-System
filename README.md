# 🎓 Student Performance Prediction & Explainable AI (XAI) System

An end-to-end Machine Learning and Interactive Web Application built with **Python & Streamlit** to predict student academic performance, identify at-risk students early, provide Explainable AI (XAI) reasoning, and recommend personalized study plans.

## 🚀 Key Features
1. **Single Student Prediction & Teacher Rating:** Predicts CGPA and Risk Level (Safe / Moderate / High Risk) incorporating academic metrics and teacher observations.
2. **Automatic Risk Warning System:** Triggers instant alerts and banners when a student falls into the High Risk/Dropout zone.
3. **Explainable AI (XAI Insights):** Visualizes positive and negative feature contributions explaining *why* a student received a specific prediction.
4. **"What-If" Interactive Simulator:** Allows students to adjust study hours, attendance, and habits via sliders to see projected CGPA improvements.
5. **Actionable Recommendations:** Automatically generates personalized study routines and exam benchmarks.
6. **Batch Class Prediction (CSV/Excel):** Enables educators to upload full class sheets and download filtered At-Risk student lists.
7. **Analytical Dashboard & Thesis Defense Comparison:** Displays EDA charts alongside Accuracy, F1-Score, and Confusion Matrices for Random Forest, XGBoost, and SVM.

## 🛠️ How to Run Locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py