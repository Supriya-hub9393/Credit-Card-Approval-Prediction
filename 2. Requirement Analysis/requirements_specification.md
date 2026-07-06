# Phase 2: System Requirement Specifications (SRS)

## 1. Functional Requirements
*   **Data Ingestion:** Must handle applicant data profiles containing features like gender, income type, annual income, employment duration, and education qualification.
*   **Feature Transformation Pipeline:** Must process multi-class loan records into structured, high-risk binary indicators.
*   **Prediction Interface:** Must host an interface to parse inputs and render clear approval/rejection outcomes alongside contextual warning markers.

## 2. Software Requirements
*   **Language Environment:** Python (v3.10+)
*   **Data Processing Libraries:** NumPy (v1.26.4), Pandas (v2.2.1)
*   **Data Visualization Engines:** Matplotlib (v3.8.3), Seaborn (v0.13.2)
*   **Machine Learning Frame Frameworks:** Scikit-Learn (v1.4.1), XGBoost (v2.0.3)
*   **Web Framework Routing:** Flask (v3.0.2)
*   **Cloud Operations & Scalability:** IBM Watson Machine Learning Deployment SDK

## 3. Hardware Requirements
*   **Development Instance:** Minimum 4GB RAM, Core i3 Processor or equivalent cloud container environment.
*   **Disk Footprint:** Less than 500MB local footprint for model serializations and web interface components.
