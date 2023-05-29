from flask import Flask, render_template, request
from passlib.hash import sha256_crypt
import mysql.connector as mysqlconn

app = Flask(__name__)

db_connection = mysqlconn.connect(user='GiuseppeVolpe', password='password', database='alive_users')

@app.route('/')
def index():
  username = "GiuVol"
  password = sha256_crypt.encrypt("newPassword")
  email = "what@ever.com"

  cur = db_connection.cursor()
  cur.execute('INSERT INTO Login (username, password, email) VALUES (%s, %s, %s)', (username, password, email))
  db_connection.commit()
  cur.close()

  return "New user added"
if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port='5000')
