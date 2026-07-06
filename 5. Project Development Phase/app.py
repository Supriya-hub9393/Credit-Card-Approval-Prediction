from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)
MODEL_PATH = os.path.join('models', 'best_model.pkl')

def get_predictive_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

@app.route('/')
def home():
    return render_template('index.html', active_scenario='scenario1', form_data={})

@app.route('/scenario/<name>')
def switch_scenario(name):
    return render_template('index.html', active_scenario=name, form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        curr = request.form.get('current_scenario', 'scenario1')
        
        # Capture form values as strings to pass back into the fields safely
        form_data = {
            'age': request.form.get('age', '35'),
            'gender': request.form.get('gender', 'Male'),
            'education_level': request.form.get('education_level', 'Graduate'),
            'marital_status': request.form.get('marital_status', 'Single'),
            'card_type': request.form.get('card_type', 'Regular'),
            'monthly_income': request.form.get('monthly_income', '50000'),
            'existing_loans_emi': request.form.get('existing_loans_emi', '12000'),
            'cibil_score': request.form.get('cibil_score', '800'),
            'city_tier': request.form.get('city_tier', 'Tier-1'),
            'employment_type': request.form.get('employment_type', 'Salaried'),
            'employment_duration': request.form.get('employment_duration', '10.0'),
            'pan_status': request.form.get('pan_status', 'Valid'),
            'previous_defaults': request.form.get('previous_defaults', 'No')
        }

        # Convert numeric items for algorithmic checks
        inc = float(form_data['monthly_income']) if form_data['monthly_income'].strip() else 50000.0
        emi = float(form_data['existing_loans_emi']) if form_data['existing_loans_emi'].strip() else 0.0
        cibil = int(form_data['cibil_score'])
        age = int(form_data['age'])
        exp = float(form_data['employment_duration'])

        risk = 1 if (cibil < 650 or form_data['pan_status'] == 'Invalid' or form_data['previous_defaults'] == 'Yes') else 0
        model = get_predictive_model()
        
        if model:
            pred = model.predict(np.array([[age, exp, inc, cibil, emi, risk]]))
        else:
            if form_data['card_type'] == 'Secured':
                pred = 0 
            elif risk == 1 or inc < 25000 or emi > (inc * 0.45):
                pred = 1
            else:
                pred = 0

        if pred == 0:
            msg, cls = "Prediction: This applicant is likely ELIGIBLE FOR APPROVAL based on risk models.", "success"
        else:
            msg, cls = "Prediction: This application is likely to be REJECTED based on risk parameters.", "danger"

        return render_template('index.html', prediction_text=msg, alert_class=cls, active_scenario=curr, form_data=form_data)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", alert_class="danger", active_scenario='scenario1', form_data={})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
