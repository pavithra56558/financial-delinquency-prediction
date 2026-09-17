Financial Delinquency Analysis & Risk Prediction
Project Overview

This repository contains an end-to-end credit risk analysis built on customer financial and repayment data, developed as part of the Tata GenAI Powered Data Analytics Job Simulation (Forage). The project investigates the key behavioral and financial drivers of loan delinquency, builds machine learning models to predict default risk, and translates the findings into an AI-driven, responsible collections strategy for a consumer credit portfolio.

Key Features & Insights
Risk Driver Analysis: Answered 9 structured business questions examining how delinquency varies by credit score, credit utilization, income, missed payments, debt-to-income ratio, loan balance, employment status, account tenure, and credit card type.
Credit Utilization as Leading Indicator: Accounts using over 80% of their available credit limit showed the sharpest jump in delinquency, reaching 26.32% — the strongest single warning sign identified in the dataset.
Income ≠ Safety: Customers earning $125K–$150K showed the highest default rate (23.40%) of any income bracket, disproving the assumption that higher income reliably lowers risk.
Missed Payments Tipping Point: Risk stayed flat (13%–18%) for 0–4 missed payments, then spiked to 25.0% at 5+ missed payments.
Product & Segment Risk: Business (21.30%) and Student (17.86%) credit cards carried the highest default rates, while Platinum cardholders (11.84%) were the most reliable segment.
Predictive Modeling: Built a Logistic Regression model to predict delinquency probability, with threshold tuning (0.45–0.60) to balance precision vs. recall for a real-world collections use case.
AI-Driven Collections Strategy: Designed a 4-stage autonomous collections architecture (Data Inputs → Predictive Logic → Targeted Actions → Learning Loop) with SHAP-based explainability, bias/fairness audits, and FDCPA/TCPA compliance guardrails.
Projected Business Impact: Modeled strategy is projected to reduce 90-day delinquencies by 20%, cut cost-to-collect by 35%, and lift promise-to-pay conversions by 15%.
Dataset Summary
Metric	Value
Total Customer Accounts Analyzed	500
Features Used in Modeling	Credit Utilization, Missed Payments, Income, Debt-to-Income Ratio, Account Tenure
Highest-Risk Segment	80–90% Credit Utilization (26.32% delinquency)
Lowest-Risk Segment	10–20% Credit Utilization (0.0% delinquency)
Business Questions Analyzed	9
Model Built	Logistic Regression
Train/Test Split	80% / 20% (Stratified)
Technologies & Tools Used
Python (Pandas, NumPy): Data cleaning, binning, and feature engineering (credit score tiers, utilization bins, income brackets, DTI tiers).
Matplotlib & Seaborn: Visualization of delinquency trends across risk segments (bar charts, pie charts, ROC curves).
Scikit-learn: Model development using LogisticRegression, with train_test_split, StandardScaler, and threshold-based evaluation (precision, recall, ROC-AUC, confusion matrix).
Joblib: Model and scaler serialization for deployment (.pkl export).
Jupyter Notebook: Interactive environment for EDA, hypothesis testing, and model iteration.
GenAI Tools: Used to synthesize findings into a business-facing executive briefing and design the autonomous collections strategy architecture.
Repository Contents
Financial_Delinquency_Analysis___Risk_Prediction.ipynb — Exploratory data analysis and risk-driver investigation.
predictive_model_Financial_Delinquency_Analysis___Risk_Prediction.ipynb — Model training, evaluation, and threshold tuning.
Autonomous_AI_Collections_Strategy_Presentation.pdf — Executive briefing translating analysis into a responsible AI collections strategy.
Notes on Model Performance

The Logistic Regression model was evaluated at multiple decision thresholds (0.45–0.60) to manage the precision-recall tradeoff typical of imbalanced delinquency data (minority class ≈16%). Model coefficients showed Income, Credit Utilization, and Debt-to-Income Ratio as positively associated with delinquency risk. Results highlighted the difficulty of predicting default from limited behavioral features alone, reinforcing the case for the layered, human-in-the-loop collections approach outlined in the strategy presentation rather than relying on model output in isolation.
