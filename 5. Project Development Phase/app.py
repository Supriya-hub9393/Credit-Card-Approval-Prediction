from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# --- CONFIGURATION PATHS ---
MODEL_PATH = os.path.join('models', 'best_model.pkl')

def get_predictive_model():
    """Attempts to load the serialized machine learning model if available."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            return pickle.load(file)
    return None

# --- WEB APPLICATON FRONTEND ROUTING ---

@app.route('/')
def home():
    """Renders the self-service web interactive input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles real-time form submissions from bank analysts or customers."""
    try:
        # 1. Extract inputs from the user web form
        annual_income = float(request.form.get('annual_income', 0.0))
        employment_duration = float(request.form.get('employment_duration', 0.0))
        gender = int(request.form.get('gender', 0))
        education_level = int(request.form.get('education_level', 0))
        income_type = int(request.form.get('income_type', 0))
        past_due_status = int(request.form.get('past_due_status', 0))

        # 2. Feature Engineering Layer (Scenario 2: Multi-class payment codes to Binary Risk Indicator)
        # 0 = Low Risk (Clean or under 30 days overdue), 1 = High Risk (Over 30 days overdue)
        is_high_risk = 1 if past_due_status >= 2 else 0

        # Assemble the clean input array matching the machine learning training layout schema
        input_features = np.array([[gender, income_type, annual_income, employment_duration, education_level, is_high_risk]])

        # 3. Model Inference Pipeline Execution
        trained_model = get_predictive_model()
        
        if trained_model:
            # Active production routing (Logistic Regression / Random Forest / XGBoost / Decision Tree)
            prediction = trained_model.predict(input_features)[0]
        else:
            # Automated fallback ruleset prior to project file uploads
            if is_high_risk == 1:
                prediction = 1  # Auto-reject: High-Risk Delinquency Infraction
            elif annual_income < 25000 and employment_duration < 1.0:
                prediction = 1  # Auto-reject: Shallow Underwriting Thresholds
            else:
                prediction = 0  # Approved: Stable credit profile

        # 4. Format Output Presentation Styling
        if prediction == 0:
            outcome_message = "✅ Application Approved! The applicant's profile satisfies bank underwriting rules."
            css_class = "success"
        else:
            outcome_message = "❌ Application Rejected. The financial risk metric exceeds acceptable variations."
            css_class = "danger"

        return render_template('index.html', prediction_text=outcome_message, alert_class=css_class)

    except Exception as error:
        return render_template('index.html', prediction_text=f"Data Preprocessing Pipeline Error: {str(error)}", alert_class="danger")


# --- IBM WATSON CLOUD PIPELINE MOCK ENDPOINT ---

@app.route('/api/watson_deploy', methods=['POST'])
def api_watson_deploy():
    """
    Placeholder endpoint mirroring an active IBM Watson Machine Learning real-time scoring request.
    Enables cloud deployment testing before binding final credentials.
    """
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({'status': 'error', 'message': 'Payload body missing fields'}), 400
            
        return jsonify({
            'status': 'connected',
            'deployment_target': 'IBM Cloud Watson WML Pipeline',
            'message': 'Real-time API cloud environment bridge verified.',
            'input_received': json_data
        })
    except Exception as api_error:
        return jsonify({'status': 'error', 'message': str(api_error)}), 500


if __name__ == '__main__':
    # Boot the localized core service deployment environment framework on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
