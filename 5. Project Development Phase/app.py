from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

MODEL_PATH = os.path.join('models', 'best_model.pkl')

def get_predictive_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            return pickle.load(file)
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        annual_income = float(request.form.get('annual_income', 0.0))
        employment_duration = float(request.form.get('employment_duration', 0.0))
        gender = int(request.form.get('gender', 0))
        education_level = int(request.form.get('education_level', 0))
        income_type = int(request.form.get('income_type', 0))
        past_due_status = int(request.form.get('past_due_status', 0))

        is_high_risk = 1 if past_due_status >= 2 else 0
        input_data = np.array([[gender, income_type, annual_income, employment_duration, education_level, is_high_risk]])

        trained_model = get_predictive_model()
        if trained_model:
            prediction = trained_model.predict(input_data)
        else:
            if is_high_risk == 1 or (annual_income < 25000 and employment_duration < 1.0):
                prediction = 1
            else:
                prediction = 0

        if prediction == 0:
            outcome_message = "✅ Application Approved! Profile meets systemic credit variances."
            css_class = "success"
        else:
            outcome_message = "❌ Application Rejected. Underwriting risk flags detected."
            css_class = "danger"

        return render_template('index.html', prediction_text=outcome_message, alert_class=css_class)

    except Exception as error:
        return render_template('index.html', prediction_text=f"Pipeline Error: {str(error)}", alert_class="danger")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
