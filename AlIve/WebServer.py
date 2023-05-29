from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort
import mysql.connector as mysqlconn
import os
import operator
from ModelsAndDatasets import *

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

    return login_form()

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
    
    user_tuple = cur.fetchone()
    
    id = user_tuple[0]
    correct_password = user_tuple[2]
    email = user_tuple[3]

    logged = (inserted_password == correct_password)
    
    if logged:
        session['logged_in'] = True
    else:
        flash('wrong password!')
    
    if request.method == 'POST':
        session["userid"] = id
        session['username'] = username
        session['useremail'] = email
    return home()

@app.route('/logout')
def logout():
    reset_session()
    return home()

@app.route('/create_env', methods=['POST'])
def create_environment():

    form = request.form
    
    env_name = form['env_name']
    
    cur = db_connection.cursor()
    
    cur.execute('SELECT env_id, env_name FROM users_environments WHERE user_id = {}'.format(session["userid"]))
    environments = cur.fetchall()

    max_id = 0

    for environment in environments:
        
        if environment[0] > max_id:
            max_id = environment[0]

        if environment[1] == env_name:
            print("An environment with this name already exists!")
            return home()
    
    new_id = max_id + 1
        
    path_to_env = "AlIve/UsersData/" + session["username"] + "/Environments/" + env_name + "/"

    try:
        if not os.path.exists(path_to_env):
            os.makedirs(path_to_env)
    except:
        print("Couldn't create environment!")
        return home()

    try:
        cur.execute("INSERT INTO users_environments (user_id, env_id, env_name) VALUES ({}, {}, '{}')"
                    .format(session['userid'], new_id, env_name) )
    except Exception as ex:
        print("Couldn't create environment! " + str(ex))
    finally:
        return home()

@app.route('/select_env', methods=['POST'])
def select_environment():

    form = request.form

    env_id = None
    env_name = None
    
    if "envid" in form:
        env_id = form['envid']
    
    if "envname" in form:
        env_name = form['envname']
    
    if env_id == None and env_name == None:
        print("No env identifier given!")
        return home()
    
    cur = db_connection.cursor()
    
    if env_id != None:
        query = "SELECT env_id, env_name FROM users_environments WHERE user_id = {} AND env_id = {}".format(session["userid"], env_id)
    elif env_name != None:
        query = "SELECT env_id, env_name FROM users_environments WHERE user_id = {} AND env_name = '{}'".format(session["userid"], env_name)
    
    cur.execute(query)
    environments = cur.fetchall()

    if len(environments) == 0:
        print("Inexisting environment!")
    else:
        session["envid"] = environments[0][0]
        session["envname"] = environments[0][1]
    
    return home()

@app.route('/select_env', methods=['POST'])
def create_model():

    form = request.form

    if "envid" not in session:
        print("No environment selected!")
        return home()
    
    if "model_name" not in form:
        print("No name given!")
    
    userid = session["userid"]
    envid = session["envid"]
    envname = session["envname"]
    model_name = "NewModel"
    model_type = "SLCM"
    finetunable = False
    base_model = ""
    num_of_classes = 7
    encoder_trainable = False
    encoder_output_key = ""
    dropout_rate = 0.1
    final_output_activation = ""
    optimizer_lr = ""
    additional_metrics = []

    encoder_link, preprocess_link = get_handle_preprocess_link(base_model)
    
    query = "SELECT * FROM environments_models WHERE user_id = {} AND env_id = {}".format(userid, envid)
    
    cur = db_connection.cursor()
    
    cur.execute(query)
    models = cur.fetchall()

    max_id = 0

    for model in models:

        if model[2] > max_id:
            max_id = model[2]
        
        if model[3] == model_name:
            print("A model with this name already exists!")
            return home()
    
    new_id = max_id + 1

    path_to_env = "AlIve/UsersData/" + session["username"] + "/Environments/" + envname + "/"
    path_to_model = path_to_env + model_name + "/"

    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)
    
    if model_type == "SLCM":
        new_model = SentenceLevelClassificationModel(model_name, finetunable)
        new_model.build(encoder_link, num_of_classes, preprocess_link, encoder_trainable, 
                        encoder_output_key, dropout_rate, final_output_activation, optimizer_lr, 
                        additional_metrics, True)
        new_model.save(path_to_model, True)
    elif model_type == "TLCM":
        new_model = TokenLevelClassificationModel(model_name, finetunable)
        new_model.build(preprocess_link, encoder_link, num_of_classes, encoder_trainable, 
                        dropout_rate=dropout_rate, optimizer_lr=optimizer_lr, additional_metrics=[])
        new_model.save(path_to_model, True)
    
    return home()

def initialize_server():

    path_to_users_data = "AlIve/UsersData/"

    if not os.path.exists(path_to_users_data):
        os.makedirs(path_to_users_data)
    
def reset_session():
    session['logged_in'] = False
    session.pop("userid")
    session.pop("username")
    session.pop("useremail")
    session.pop("envid")
    session.pop("envname")

if __name__ == "__main__":
    initialize_server()
    app.secret_key = os.urandom(12)
    app.run(debug=False,host='0.0.0.0', port=5000)
