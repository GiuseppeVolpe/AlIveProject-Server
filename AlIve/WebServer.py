from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort
import mysql.connector as mysqlconn
import os
import operator

app = Flask(__name__)
db_connection = mysqlconn.connect(user='GiuseppeVolpe', password='password', database='alive_db')

@app.route('/')
def home():
  if not session.get('logged_in'):
    return render_template('login.html')
  else:
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def add_user():

    form = request.form

    username = form['username']
    password = form['password']
    email = form['email']

    cur = db_connection.cursor()
    
    try:
        cur.execute('INSERT INTO alive_users (username, user_password, email) VALUES (%s, %s, %s)', (username, password, email))
        db_connection.commit()
        cur.close()
    except Exception as ex:
        print(ex)
        return("Couldn't add user...")

    return "New user added"

@app.route('/login', methods=['POST'])
def do_admin_login():
  login = request.form

  userName = login['username']
  password = login['password']

  cur = db_connection.cursor(buffered=True)
  data = cur.execute('SELECT * FROM alive_users WHERE username=%s', (userName))
  data = cur.fetchone()[2]

  if password == data:
    account = True

  if account:
    session['logged_in'] = True
  else:
    flash('wrong password!')
  return home()

@app.route('logout')
def logout():
  session['logged_in'] = False
  return home()

if __name__ == "__main__":
  app.secret_key = os.urandom(12)
  app.run(debug=False,host='0.0.0.0', port=5000)