import psycopg2  # pip install psycopg2
import re
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2.extras
from flask import Flask, request, session, redirect, url_for, render_template, flash
from flask import Flask, render_template, request, flash
# generate random integer values
from random import randint
from datetime import datetime
# import datetime
import datetime as dt
import trafficDetection

app = Flask(__name__)
app.secret_key = 'hi'

DB_HOST = "localhost"
DB_NAME = "traffic"
DB_USER = "postgres"
DB_PASS = "2000"

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER,
                        password=DB_PASS, host=DB_HOST)


@app.route('/')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home.html', name=session['name'])
    # User is not loggedin redirect to login page
    return redirect(url_for('logout'))



@app.route('/login/', methods=['GET', 'POST'])
def login():
    cursor = conn.cursor()
    cursor.execute("ROLLBACK")
    conn.commit()

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Check if "email_id" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'email_id' in request.form and 'password' in request.form:
        email_id = request.form['email_id']
        password = request.form['password']

        # Check if account exists using MySQL
        cursor.execute(
            'SELECT * FROM members WHERE email_id = %s', (email_id,))
        # Fetch one record and return result
        account = cursor.fetchone()
   

        if account:
            password_rs = account['password']
            # If account exists in users table in out database
            if password_rs == password:
                session['loggedin'] = True
                session['email_id'] = account['email_id']
                session['name'] = account['name']
                # Redirect to home page
                return redirect(url_for('home'))
            else:
                # Account doesnt exist or email_id/password incorrect
                flash('Incorrect inner email_id/password')
        else:
            # Account doesnt exist or email_id/password incorrect
            flash('Incorrect email_id/password')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    cursor = conn.cursor()
    cursor.execute("ROLLBACK")
    conn.commit()

    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Check if "email_id", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'email_id' in request.form and 'phone_number' in request.form and 'phone_number2' in request.form and 'gender' in request.form and 'password' in request.form:
        # Create variables for easy access
        fullname = request.form['fullname']
        email_id = request.form['email_id']
        password = request.form['password']
        phone_number = request.form['phone_number']
        phone_number2 = request.form['phone_number2']
        gender = request.form['gender']
        passwordopen = request.form['password']
        cursor = conn.cursor()
        cursor.execute("ROLLBACK")
        conn.commit()

        # Check if account exists using MySQL
        cursor.execute(
            'SELECT * FROM members WHERE email_id = %s', (email_id,))
        account = cursor.fetchone()

        if account:
            flash('Account already exists!')
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email_id):
            flash('Invalid email address!')
        elif not re.match(r'[A-Za-z0-9]+', email_id):
            flash('email_id must contain only characters and numbers!')
        elif not email_id or not password or not email_id:
            flash('Please fill out the form!')
        else:
            # Account doesnt exists and the form data is valid, now insert new account into users table
            lane_sound = 'Yes'
            traffic_sound ='No'
            road_lanes ='Yes'
            traffic_sign ='No'
            cursor.execute("INSERT INTO members (email_id, name, password, gender ) VALUES (%s,%s,%s,%s)",
                           (email_id, fullname,  passwordopen, gender,))
            cursor.execute("INSERT INTO preference (lane_sound ,traffic_sound ,road_lanes ,traffic_sign ,email_id) VALUES (%s,%s,%s,%s,%s)",
                           (lane_sound ,traffic_sound ,road_lanes ,traffic_sign ,email_id,))
            cursor.execute("INSERT INTO phoneno (p_no , p_no2 ,email_id) VALUES (%s,%s,%s)",
                           (phone_number, phone_number2, email_id,))
            conn.commit()
            flash('You have successfully registered!')
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash('Please fill out the form!')
    # Show registration form with message (if any)
    return render_template('register.html')


@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
    if 'loggedin' in session:
        session.pop('loggedin', None)
        # session.pop('start_station', None)
        session.pop('email_id', None)
        session.pop('name', None)
        cursor = conn.cursor()
        cursor.execute("ROLLBACK")
        conn.commit()
        # Redirect to login page

        return redirect(url_for('login'))
    return redirect(url_for('main'))


@app.route('/profile')
def profile():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Check if user is loggedin
    if 'loggedin' in session:
        cursor.execute('SELECT * FROM members WHERE email_id = %s',
                       [session['email_id']])
        account = cursor.fetchone()
        cursor.execute('SELECT * FROM phoneno WHERE email_id = %s',
                       [session['email_id']])
        phoneno = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account, phoneno1=phoneno['p_no'], phoneno2=phoneno['p_no2'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Check if user is loggedin
    if 'loggedin' in session:
        cur_email_id = session['email_id']
        cursor.execute('SELECT * FROM members WHERE email_id = %s',
                       (cur_email_id,))
        account = cursor.fetchone()
        cursor.execute('SELECT * FROM phoneno WHERE email_id = %s',
                       (cur_email_id,))
        phoneno = cursor.fetchone() 
        # Show the profile page with account info
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        if request.method == 'POST' and 'subject' in request.form and 'report_con' in request.form:
            # Create variables for easy access
            subject = request.form['subject']
            feedback_content = request.form['report_con']
            cursor.execute("INSERT INTO feedback (feedback_subject , feedback ,email_id) VALUES (%s,%s,%s)",
                           (subject, feedback_content, session['email_id'],))
            conn.commit()
            flash('You have successfully gave your feedback!')
            return render_template('feedback.html', account=account, phoneno1=phoneno['p_no'], phoneno2=phoneno['p_no2'])
        elif request.method == 'POST':
            # Form is empty... (no POST data)
            flash('outer Please fill out the form!')
        return render_template('feedback.html', account=account, phoneno1=phoneno['p_no'], phoneno2=phoneno['p_no2'])
    # User is not loggedin redirect to login page
    return render_template(url_for('login'))


@app.route('/main', methods=['GET', 'POST'])
def main():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if 'loggedin' in session:
        cursor.execute('SELECT * FROM preference WHERE email_id = %s', [session['email_id']])
        preference = cursor.fetchone()
        lane_sound = preference['lane_sound']
        traffic_sound = preference['traffic_sound']
        road_lanes = preference['road_lanes']
        traffic_sign = preference['traffic_sign']
        email_id = preference['email_id']
        trafficDetection.solve(lane_sound ,traffic_sound ,road_lanes ,traffic_sign)
        return redirect(url_for('detect'))
    # Show registration form with message (if any)
    return render_template('login.html')

    ##################################################


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if not ('loggedin' in session):
        return redirect(url_for('login'))
    
    if 'loggedin' in session:

        cursor.execute('SELECT * FROM preference WHERE email_id = %s', [session['email_id']])
        preference = cursor.fetchone()
        lane_sound = preference['lane_sound']
        traffic_sound = preference['traffic_sound']
        road_lanes = preference['road_lanes']
        traffic_sign = preference['traffic_sign']
        email_id = preference['email_id']
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        if request.method == 'POST' and 'traffic_sign' in request.form and 'traffic_sound' in request.form and 'road_lanes' in request.form  and 'lane_sound' in request.form:
            # Create variables for easy access
            traffic_sign = str(request.form['traffic_sign'])
            traffic_sound = str(request.form['traffic_sound'])
            road_lanes = str(request.form['road_lanes'])
            lane_sound = str(request.form['lane_sound'])
            cursor.execute('UPDATE preference SET (lane_sound ,traffic_sound ,road_lanes ,traffic_sign)=(%s,%s,%s,%s) WHERE email_id=%s',
                           (lane_sound ,traffic_sound ,road_lanes ,traffic_sign,session['email_id'],))

            conn.commit()
            flash('You have successfully started detecting!')
            # lane_sound ,tarffic_sound ,road_lanes ,traffic_sign ,email_id
            trafficDetection.solve(lane_sound ,traffic_sound ,road_lanes ,traffic_sign)
            return render_template('detect.html', lane_sound=lane_sound, traffic_sound=traffic_sound, road_lanes=road_lanes, traffic_sign=traffic_sign,email_id=email_id,)

        # return render_template('detect.html', lane_sound=lane_sound, traffic_sound=traffic_sound, road_lanes=road_lanes, traffic_sign=traffic_sign,)
                # return render_template('detect.html', metro_id=metro_id, metro=metro, metro_card=metro_card, station=station, boards=boards, gender=gender, waiting_time=waiting_time,)
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        flash('outer Please fill out the form!')
    # Show registration form with message (if any)
    return render_template('detect.html', lane_sound=lane_sound, traffic_sound=traffic_sound, road_lanes=road_lanes, traffic_sign=traffic_sign, email_id=email_id,)
    # return render_template('detect.html', lane_sound=, tarffic_sound='No', road_lanes='No', traffic_sign='Yes',)
    # return render_template('detect.html')
    # return render_template('detect.html', metro_card=metro_card, station=station, metro=metro, waiting_time=waiting_time, )


if __name__ == "__main__":
    app.run(debug=True)
