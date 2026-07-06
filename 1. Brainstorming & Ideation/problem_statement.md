# Phase 1: Brainstorming & Ideation - Credit Card Approval Prediction

## 1. Problem Statement & Context
Financial institutions receive an overwhelming volume of credit card applications daily. Manually reviewing applications is time-consuming, expensive, and vulnerable to subjective human error. This project uses classification algorithms to build an automated, reliable risk underwriting pipeline.

## 2. Target User Personas & Real-World Scenarios

### Scenario 1: Bank Credit Analyst Screening
*   **User:** Bank Credit Analyst.
*   **Workflow:** Inputs financial demographics directly into an online evaluation interface. The system delivers instantaneous approval decisions, isolating borderline applications for manual review.

### Scenario 2: Compliance Officer High-Risk Screening
*   **User:** Financial Compliance and Risk Management Officer.
*   **Workflow:** Runs automated data parsing to transform multi-class payment delays into strict binary default metrics, filtering high-risk portfolios instantly.

### Scenario 3: Customer Self-Service Eligibility Check
*   **User:** Prospective Retail Bank Customer.
*   **Workflow:** Checks credit card eligibility through a self-service page before submitting a formal application, preventing unnecessary rejection records.

## 3. Machine Learning Objectives
*   Develop an automated prediction framework using **Logistic Regression, Decision Trees, Random Forests, and XGBoost**.
*   Select the champion architecture with the highest test accuracy to deploy real-time routing endpoints via Flask.
