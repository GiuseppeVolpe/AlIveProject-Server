from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort
import mysql.connector as mysqlconn
import os
import operator

app = Flask(__name__, template_folder='Templates')
db_connection = mysqlconn.connect(user='GiuseppeVolpe', password='password', database='alive_db')

@app.route('/')
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return render_template('index.html')

@app.route('/signup_form', methods=['POST'])
def signup_form():
    return render_template('signup.html')

@app.route('/login_form')
def login_form():
    return render_template('login.html')

@app.route('/user_space', methods=['POST'])
def user_space():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():

    form = request.form

    username = form['username']
    password = form['password']
    email = form['email']

    if len(username) < 2:
        flash("This username is too short")
        return signup_form()
    
    if len(password) < 8:
        flash("The length of the password should be at least 8")
        return signup_form()
    
    cur = db_connection.cursor()
    
    cur.execute('SELECT username FROM alive_users WHERE username="{}"'.format(username))
    usernames = cur.fetchall()

    print(len(usernames))

    if len(usernames) > 0:
        flash("This username is already taken!")
        return signup_form()
    
    cur.execute('SELECT email FROM alive_users WHERE email="{}"'.format(email))
    email_addresses = cur.fetchall()

    print(len(email_addresses))

    if len(email_addresses) > 0:
        flash("This email address is already taken!")
        return signup_form()
    
    try:
        cur.execute('INSERT INTO alive_users (username, user_password, email) VALUES (%s, %s, %s)', (username, password, email))
        db_connection.commit()
        cur.close()
    except Exception as ex:
        print(ex)
        flash("Couldn't add user...")
        return signup_form()
    
    user_space_path = "AlIve/UsersData/" + username + "/"
    user_environments_path = user_space_path + "/Environments/"

    if not os.path.exists(user_environments_path):
        os.makedirs(user_environments_path)

    return "New user added"

@app.route('/login', methods=['POST'])
def login():

    login = request.form

    username = login['username']
    inserted_password = login['password']

    cur = db_connection.cursor(buffered=True)
    cur.execute('SELECT * FROM alive_users WHERE username = "{}"'.format(username))

    if cur.rowcount == 0:
        flash("This user doesn't exist!")
        return home()
    
    correct_password = cur.fetchone()[2]

    if inserted_password == correct_password:
        logged = True
    else:
        logged = False

    if logged:
        session['logged_in'] = True
    else:
        flash('wrong password!')
    
    return home()

@app.route('/logout')
def logout():
    session['logged_in'] = False
    return home()

def initialize_server():

    path_to_users_data = "AlIve/UsersData/"

    if not os.path.exists(path_to_users_data):
        os.makedirs(path_to_users_data)
    
if __name__ == "__main__":
    initialize_server()
    app.secret_key = os.urandom(12)
    app.run(debug=False,host='0.0.0.0', port=5000)
