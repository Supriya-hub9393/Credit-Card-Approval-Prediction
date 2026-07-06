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
    return render_template('index.html', active_scenario='scenario1')

@app.route('/scenario/<name>')
def switch_scenario(name):
    return render_template('index.html', active_scenario=name)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        curr = request.form.get('current_scenario', 'scenario1')
        age = int(request.form.get('age', 35))
        
        inc = request.form.get('monthly_income', '50000')
        inc = float(inc) if inc.strip() else 50000.0
        
        emi = request.form.get('existing_loans_emi', '0')
        emi = float(emi) if emi.strip() else 0.0
        
        cibil = int(request.form.get('cibil_score', 750))
        exp = float(request.form.get('employment_duration', 3.0))
        pan = request.form.get('pan_status', 'Valid')
        default = request.form.get('previous_defaults', 'No')
        card_type = request.form.get('card_type', 'Regular')

        risk = 1 if (cibil < 650 or pan == 'Invalid' or default == 'Yes') else 0
        model = get_predictive_model()
        
        if model:
            pred = model.predict(np.array([[age, exp, inc, cibil, emi, risk]]))
        else:
            if card_type == 'Secured':
                pred = 0 
            elif risk == 1 or inc < 25000 or emi > (inc * 0.45):
                pred = 1
            else:
                pred = 0

        if pred == 0:
            msg, cls = "Prediction: This applicant is likely ELIGIBLE FOR APPROVAL based on risk models.", "success"
        else:
            msg, cls = "Prediction: This application is likely to be REJECTED based on risk parameters.", "danger"

        return render_template('index.html', prediction_text=msg, alert_class=cls, active_scenario=curr)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", alert_class="danger", active_scenario='scenario1')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
