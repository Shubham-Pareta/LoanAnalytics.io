from flask import Flask, render_template, request
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import mysql.connector

app = Flask(__name__)

# Load the scaler and the model
scaler = joblib.load('models/scaler.lb')
model = load_model('./models/ann_model.h5')

# MySQL Database Configuration
db_config = {
    'user': 'root',
    'password': '@Shubham2003',
    'host': 'localhost',
    'database': 'loan_approval'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/basic")
def basic_page():
    return render_template('basic.html')

@app.route("/project")
def project_page():
    return render_template('project.html')

@app.route('/prediction', methods=['GET', 'POST'])
def make_prediction():
    if request.method == 'POST':
        # Extract form data preserving the original data types
        no_of_dependents = request.form['no_of_dependents']
        income_annum = request.form['income_annum']
        loan_amount = request.form['loan_amount']
        loan_term = request.form['loan_term']
        cibil_score = request.form['cibil_score']
        residential_assets_value = request.form['residential_assets_value']
        commercial_assets_value = request.form['commercial_assets_value']
        luxury_assets_value = request.form['luxury_assets_value']
        bank_asset_value = request.form['bank_asset_value']
        education_not_graduate = request.form['education_Not_Graduate']
        self_employed_yes = request.form['self_employed_Yes']
        
        # Preprocess the data for the model
        data = np.array([
            int(no_of_dependents),
            float(income_annum),
            float(loan_amount),
            int(loan_term),
            int(cibil_score),
            float(residential_assets_value),
            float(commercial_assets_value),
            float(luxury_assets_value),
            float(bank_asset_value),
            int(education_not_graduate),
            int(self_employed_yes)
        ]).reshape(1, -1)
        
        data_scaled = scaler.transform(data)
        
        # Make a prediction
        prediction = model.predict(data_scaled)
        output = 'Loan Approved' if prediction[0][0] >= 0.5 else 'Loan Rejected'
        
        # Insert the data and prediction result into the MySQL database
        try:
            connection = mysql.connector.connect(**db_config)
            cursor = connection.cursor()
            insert_query = """
                INSERT INTO predictions (
                    no_of_dependents, income_annum, loan_amount, loan_term, cibil_score, 
                    residential_assets_value, commercial_assets_value, luxury_assets_value, 
                    bank_asset_value, education_not_graduate, self_employed_yes, prediction
                ) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                no_of_dependents, income_annum, loan_amount, loan_term, cibil_score, 
                residential_assets_value, commercial_assets_value, luxury_assets_value, 
                bank_asset_value, education_not_graduate, self_employed_yes, output
            ))
            connection.commit()
            cursor.close()
            connection.close()
        except mysql.connector.Error as err:
            print(f"Error: {err}")
        
        # Render the output page with the result
        return render_template('output.html', output=output)
    
    return render_template('project.html')

@app.route('/home')
def go_home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
