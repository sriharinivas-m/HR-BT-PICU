from flask import Flask, request, render_template, send_file, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
import re
import sqlite3

app = Flask(__name__)

model = joblib.load("Models/XGBoost.sav")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age_months = request.form.get('age_months')
        age_group = request.form.get('age_group')
        body_temperature = request.form.get('body_temperature')
        temperature_site = request.form.get('temperature_site')
        measurement_type = request.form.get('measurement_type')
        
        # Convert to float for model
        input_features = [float(x) for x in request.form.values()]
        X = [np.array(input_features)]

        # Predict heart rate and body temperature relationship
        preds = model.predict(X)[0]
        
        # Age group mapping
        age_group_map = {
            '0': 'Child',
            '1': 'Infant',
            '2': 'Newborn',
            '3': 'Teenager',
            '4': 'Toddler'
        }
        
        # Temperature site mapping
        temp_site_map = {
            '0': 'Axillary',
            '1': 'Esophageal',
            '2': 'Rectal'
        }
        
        # Measurement type mapping
        measurement_map = {
            '0': 'Continuous',
            '1': 'Manual'
        }

        return render_template("result.html",
                               result=preds,
                               age_months=age_months,
                               age_group=age_group_map.get(age_group, 'Unknown'),
                               body_temperature=body_temperature,
                               temperature_site=temp_site_map.get(temperature_site, 'Unknown'),
                               measurement_type=measurement_map.get(measurement_type, 'Unknown'))
    except Exception as e:
        # Get form data for error display
        age_months = request.form.get('age_months', 'N/A')
        age_group = request.form.get('age_group', '')
        body_temperature = request.form.get('body_temperature', 'N/A')
        temperature_site = request.form.get('temperature_site', '')
        measurement_type = request.form.get('measurement_type', '')
        
        # Age group mapping
        age_group_map = {
            '0': 'Child',
            '1': 'Infant',
            '2': 'Newborn',
            '3': 'Teenager',
            '4': 'Toddler'
        }
        
        # Temperature site mapping
        temp_site_map = {
            '0': 'Axillary',
            '1': 'Esophageal',
            '2': 'Rectal'
        }
        
        # Measurement type mapping
        measurement_map = {
            '0': 'Continuous',
            '1': 'Manual'
        }
        
        return render_template("result.html",
                               result=str(e),
                               error=True,
                               age_months=age_months,
                               age_group=age_group_map.get(age_group, 'Unknown'),
                               body_temperature=body_temperature,
                               temperature_site=temp_site_map.get(temperature_site, 'Unknown'),
                               measurement_type=measurement_map.get(measurement_type, 'Unknown'))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    else:
        username = request.form.get('user','')
        name = request.form.get('name','')
        email = request.form.get('email','')
        number = request.form.get('mobile','')
        password = request.form.get('password','')

        # Server-side validation
        username_pattern = r'^.{6,}$'
        name_pattern = r'^[A-Za-z ]{3,}$'
        email_pattern = r'^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$'
        mobile_pattern = r'^[6-9][0-9]{9}$'
        password_pattern = r'^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$'

        if not re.match(username_pattern, username):
            return render_template("signup.html", message="Username must be at least 6 characters.")
        if not re.match(name_pattern, name):
            return render_template("signup.html", message="Full Name must be at least 3 letters, only letters and spaces allowed.")
        if not re.match(email_pattern, email):
            return render_template("signup.html", message="Enter a valid email address.")
        if not re.match(mobile_pattern, number):
            return render_template("signup.html", message="Mobile must start with 6-9 and be 10 digits.")
        if not re.match(password_pattern, password):
            return render_template("signup.html", message="Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.")

        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (username,))
        if cur.fetchone():
            con.close()
            return render_template("signup.html", message="Username already exists. Please choose another.")
        
        cur.execute("insert into `info` (`user`,`name`, `email`,`mobile`,`password`) VALUES (?, ?, ?, ?, ?)",(username,name,email,number,password))
        con.commit()
        con.close()
        return redirect(url_for('login'))

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "GET":
        return render_template("signin.html")
    else:
        mail1 = request.form.get('user','')
        password1 = request.form.get('password','')
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
        data = cur.fetchone()

        if data == None:
            return render_template("signin.html", message="Invalid username or password.")    

        elif mail1 == 'admin' and password1 == 'admin':
            return render_template("home.html")

        elif mail1 == str(data[0]) and password1 == str(data[1]):
            return render_template("home.html")
        else:
            return render_template("signin.html", message="Invalid username or password.")

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/graphs')
def graphs():
	return render_template('graphs.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')




@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
