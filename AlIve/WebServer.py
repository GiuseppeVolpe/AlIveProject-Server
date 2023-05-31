import os
import re

import mysql.connector as mysqlconn
from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort

from ModelsAndDatasets import *

#region CONSTS

LOGGED_IN_FIELD_NAME = "logged_in"
USER_ID_FIELD_NAME = "user_id"
USERNAME_FIELD_NAME = "username"
USER_PASSWORD_FIELD_NAME = "user_password"
USER_EMAIL_FIELD_NAME = "user_email"
ENV_ID_FIELD_NAME = "env_id"
ENV_NAME_FIELD_NAME = "env_name"
PUBLIC_FIELD_NAME = "public"
MODEL_ID_FIELD_NAME = "model_id"
MODEL_NAME_FIELD_NAME = "model_name"
MODEL_PATH_FIELD_NAME = "model_path"
MODEL_TYPE_FIELD_NAME = "model_type"
DATASET_ID_FIELD_NAME = "dataset_id"
DATASET_NAME_FIELD_NAME = "dataset_name"
DATASET_PATH_FIELD_NAME = "dataset_path"
DATASET_TYPE_FIELD_NAME = "dataset_type"

ALIVE_DB_NAME = "alive_db"
ALIVE_DB_ADMIN_USERNAME = "GiuseppeVolpe"
ALIVE_DB_ADMIN_PASSWORD = "password"
ALIVE_DB_USERS_TABLE_NAME = "alive_users"
ALIVE_DB_ENVIRONMENTS_TABLE_NAME = "users_environments"
ALIVE_DB_MODELS_TABLE_NAME = "environments_models"
ALIVE_DB_DATASETS_TABLE_NAME = "environments_datasets"

STR_TYPE_NAME = "str"
INT_TYPE_NAME = "int"
BOOL_TYPE_NAME = "bool"

FINETUNABLE_FIELD_NAME = "finetunable"
BASEMODEL_FIELD_NAME = "base_model"
NUM_OF_CLASSES_FIELD_NAME = "num_of_classes"
ENCODER_TRAINABLE_FIELD_NAME = "encoder_trainable"
DROPOUT_RATE_FIELD_NAME = "dropout_rate"
OPTIMIZER_LR_FIELD_NAME = "optimizer_lr"

SLCM_MODEL_TYPE = "SLCM"
TLCM_MODEL_TYPE = "TLCM"

SENTENCE_TO_PREDICT_FIELD_NAME = "sent"

ROOT_FOLDER = "AlIve"
USERS_DATA_FOLDER_NAME = "UsersData"
ENVIRONMENTS_FOLDER_NAME = "Environments"
MODELS_FOLDER_NAME = "Models"
DATASETS_FOLDER_NAME = "Datasets"

USERS_DATA_FOLDER = ROOT_FOLDER + "/" + USERS_DATA_FOLDER_NAME + "/"

#endregion

app = Flask(__name__, template_folder='Templates')
db_connection = mysqlconn.connect(user=ALIVE_DB_ADMIN_USERNAME, password=ALIVE_DB_ADMIN_PASSWORD, database=ALIVE_DB_NAME)

#region FORMS GETTERS

@app.route('/')
def home():
    if not session.get(LOGGED_IN_FIELD_NAME):
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

#endregion

@app.route('/signup', methods=['POST'])
def signup():

    form = request.form

    needed_fields = [USERNAME_FIELD_NAME, USER_PASSWORD_FIELD_NAME, USER_EMAIL_FIELD_NAME]

    for needed_field in needed_fields:
        if needed_field not in form:
            return home()
    
    username = form[USERNAME_FIELD_NAME]
    user_email = form[USER_EMAIL_FIELD_NAME]
    user_password = form[USER_PASSWORD_FIELD_NAME]

    error_found = False

    if len(username) < 2:
        flash("This username is too short!")
        error_found = True
    
    email_regex = "^[a-zA-Z0-9][a-zA-Z0-9.!#$%&'*+-/=?^_`{|}~]*?[a-zA-Z0-9._-]?@[a-zA-Z0-9][a-zA-Z0-9._-]*?[a-zA-Z0-9]?\\.[a-zA-Z]{2,63}$"
    
    if not bool( re.match(email_regex, user_email) ):
        flash("This is not a valid email!")
        error_found = True
    
    if len(user_password) < 8:
        flash("The length of the password should be at least 8!")
        error_found = True
    
    if error_found:
        return signup_form()
    
    usernames = select_from_db(ALIVE_DB_USERS_TABLE_NAME, 
                               [USERNAME_FIELD_NAME], 
                               [USERNAME_FIELD_NAME], 
                               [username])
    
    if len(usernames) > 0:
        flash("This username is already taken!")
        error_found = True
    
    email_addresses = select_from_db(ALIVE_DB_USERS_TABLE_NAME, 
                                     [USER_EMAIL_FIELD_NAME], 
                                     [USER_EMAIL_FIELD_NAME], 
                                     [user_email])
    
    if len(email_addresses) > 0:
        flash("This email address is already taken!")
        error_found = True
    
    if error_found:
        return signup_form()
    
    try:
        insert_into_db(ALIVE_DB_USERS_TABLE_NAME, 
                       [USERNAME_FIELD_NAME, USER_PASSWORD_FIELD_NAME, USER_EMAIL_FIELD_NAME],
                       [username, user_password, user_email])
    except Exception as ex:
        print(ex)
        flash("Couldn't add user...")
        return signup_form()
    
    user_space_path = USERS_DATA_FOLDER + username + "/"
    user_environments_path = user_space_path + "/" + ENVIRONMENTS_FOLDER_NAME + "/"

    if not os.path.exists(user_environments_path):
        os.makedirs(user_environments_path)

    return login_form()

@app.route('/login', methods=['POST'])
def login():

    form = request.form
    
    needed_fields = [USERNAME_FIELD_NAME, USER_PASSWORD_FIELD_NAME]

    for needed_field in needed_fields:
        if needed_field not in form:
            return login_form()
    
    username = form[USERNAME_FIELD_NAME]
    inserted_password = form[USER_PASSWORD_FIELD_NAME]

    users = select_from_db(ALIVE_DB_USERS_TABLE_NAME, 
                           ["*"], 
                           [USERNAME_FIELD_NAME], 
                           [username])
    
    if len(users) == 0:
        flash("This user doesn't exist!")
        return login_form()
    
    user_tuple = users[0]
    
    user_id = user_tuple[0]
    correct_password = user_tuple[2]
    user_email = user_tuple[3]

    logged = (inserted_password == correct_password)
    
    if logged:
        session[LOGGED_IN_FIELD_NAME] = True
    else:
        flash('wrong password!')
    
    session[USER_ID_FIELD_NAME] = user_id
    session[USERNAME_FIELD_NAME] = username
    session[USER_EMAIL_FIELD_NAME] = user_email
    
    return home()

@app.route('/logout')
def logout():
    reset_session()
    return home()

@app.route('/create_env', methods=['POST'])
def create_environment():
    
    form = request.form

    needed_session_fields = [USER_ID_FIELD_NAME, USERNAME_FIELD_NAME]
    needed_form_fields = [ENV_NAME_FIELD_NAME]
    
    needed_fields_recieved = True

    for needed_session_field in needed_session_fields:
        if needed_session_field not in session:
            needed_fields_recieved = False
    
    for needed_form_field in needed_form_fields:
        if needed_form_field not in form:
            needed_fields_recieved = False
    
    if not needed_fields_recieved:
        return home()
    
    user_id = session[USER_ID_FIELD_NAME]
    username = session[USERNAME_FIELD_NAME]
    env_name = form[ENV_NAME_FIELD_NAME]
    
    if len(env_name) <= 1:
        print("Invaild name!")
        return home()
    
    environments = select_from_db(ALIVE_DB_ENVIRONMENTS_TABLE_NAME, 
                                  [ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                                  [USER_ID_FIELD_NAME], 
                                  [user_id])
    
    max_env_id = 0

    for environment in environments:
        
        if environment[0] > max_env_id:
            max_env_id = environment[0]

        if environment[1] == env_name:
            print("An environment with this name already exists!")
            return home()
    
    new_env_id = max_env_id + 1

    path_to_env = USERS_DATA_FOLDER + username + "/" + ENVIRONMENTS_FOLDER_NAME + "/" + env_name + "/"

    try:
        if not os.path.exists(path_to_env):
            os.makedirs(path_to_env)
    except:
        print("Couldn't create the environment!")
        return home()
    
    try:
        insert_into_db(ALIVE_DB_ENVIRONMENTS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                       [user_id, new_env_id, env_name])
    except Exception as ex:
        print("Couldn't create environment! " + str(ex))
    finally:
        return home()

@app.route('/delete_env', methods=['POST'])
def delete_environment():
    
    form = request.form

    if USER_ID_FIELD_NAME not in session or ENV_NAME_FIELD_NAME not in form:
        return home()
    
    user_id = session[USER_ID_FIELD_NAME]
    username = session[USERNAME_FIELD_NAME]
    env_name = form[ENV_NAME_FIELD_NAME]
    
    environments = select_from_db(ALIVE_DB_ENVIRONMENTS_TABLE_NAME, 
                                  [ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                                  [USER_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                                  [user_id, env_name])
    
    if len(environments) == 0:
        print("An environment with this name doesn't exist!")
        return home()
    
    try:
        delete_from_db(ALIVE_DB_ENVIRONMENTS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                       [user_id, env_name])
        
        shutil.rmtree(USERS_DATA_FOLDER + "/" + username + "/" + ENVIRONMENTS_FOLDER_NAME + "/" + env_name + "/")
    except:
        print("Couldn't delete the environment!")
    
    return home()

@app.route('/select_env', methods=['POST'])
def select_environment():

    form = request.form

    env_id = None
    env_name = None
    
    if ENV_ID_FIELD_NAME in form:
        env_id = form[ENV_ID_FIELD_NAME]
    
    if ENV_NAME_FIELD_NAME in form:
        env_name = form[ENV_NAME_FIELD_NAME]
    
    if env_id == None and env_name == None:
        print("No env identifier given!")
        return home()
    
    if env_id != None:
        environments = select_from_db(ALIVE_DB_ENVIRONMENTS_TABLE_NAME, 
                                      [ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                                      [ENV_ID_FIELD_NAME], 
                                      [env_id])
    elif env_name != None:
        environments = select_from_db(ALIVE_DB_ENVIRONMENTS_TABLE_NAME, 
                                      [ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                                      [ENV_NAME_FIELD_NAME], 
                                      [env_name])
    
    if len(environments) == 0:
        print("Inexisting environment!")
    else:
        session[ENV_ID_FIELD_NAME] = environments[0][0]
        session[ENV_NAME_FIELD_NAME] = environments[0][1]
    
    return home()

@app.route('/create_model', methods=['POST'])
def create_model():
    
    form = request.form

    needed_session_fields = [USER_ID_FIELD_NAME, USERNAME_FIELD_NAME, ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME]
    needed_form_fields = [MODEL_NAME_FIELD_NAME, MODEL_TYPE_FIELD_NAME, BASEMODEL_FIELD_NAME, 
                          NUM_OF_CLASSES_FIELD_NAME, DROPOUT_RATE_FIELD_NAME, OPTIMIZER_LR_FIELD_NAME]
    
    needed_fields_recieved = True

    for needed_session_field in needed_session_fields:
        if needed_session_field not in session:
            needed_fields_recieved = False
    
    for needed_form_field in needed_form_fields:
        if needed_form_field not in form:
            needed_fields_recieved = False
    
    if not needed_fields_recieved:
        return home()
    
    user_id = session[USER_ID_FIELD_NAME]
    username = session[USERNAME_FIELD_NAME]
    envid = session[ENV_ID_FIELD_NAME]
    env_name = session[ENV_NAME_FIELD_NAME]
    model_name = form[MODEL_NAME_FIELD_NAME]
    model_type = form[MODEL_TYPE_FIELD_NAME]
    finetunable = FINETUNABLE_FIELD_NAME in form
    base_model = form[BASEMODEL_FIELD_NAME]
    num_of_classes = form[NUM_OF_CLASSES_FIELD_NAME]
    encoder_trainable = ENCODER_TRAINABLE_FIELD_NAME in form
    dropout_rate = float(form[DROPOUT_RATE_FIELD_NAME])
    optimizer_lr = float(form[OPTIMIZER_LR_FIELD_NAME])
    additional_metrics = []
    public = PUBLIC_FIELD_NAME in form
    
    encoder_link, preprocess_link = get_handle_preprocess_link(base_model)
    
    models = select_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                           ["*"], 
                           [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME], 
                           [user_id, envid])
    
    max_id = 0

    for model in models:

        if model[2] > max_id:
            max_id = model[2]
        
        if model[3] == model_name:
            print("A model with this name already exists!")
            return home()
    
    new_id = max_id + 1

    path_to_env = USERS_DATA_FOLDER + username + "/" + ENVIRONMENTS_FOLDER_NAME + "/" + env_name + "/"
    path_to_model = path_to_env + "/" + MODELS_FOLDER_NAME + "/" + model_name + "/"
    
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)
    
    if model_type == SLCM_MODEL_TYPE:
        new_model = SentenceLevelClassificationModel(model_name, finetunable)
        new_model.build(encoder_link, num_of_classes, preprocess_link, encoder_trainable, "pooled_output", 
                        dropout_rate=dropout_rate, optimizer_lr=optimizer_lr, 
                        additional_metrics=additional_metrics)
        new_model.save(path_to_model, True)
    elif model_type == TLCM_MODEL_TYPE:
        new_model = TokenLevelClassificationModel(model_name, finetunable)
        new_model.build(preprocess_link, encoder_link, num_of_classes, encoder_trainable, 
                        dropout_rate=dropout_rate, optimizer_lr=optimizer_lr, additional_metrics=additional_metrics)
        new_model.save(path_to_model, True)
    
    insert_into_db(ALIVE_DB_MODELS_TABLE_NAME, 
                   [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME, 
                    MODEL_PATH_FIELD_NAME, MODEL_TYPE_FIELD_NAME, PUBLIC_FIELD_NAME], 
                   [user_id, envid, new_id, model_name, path_to_model, model_type, public])
    
    return home()

@app.route('/delete_model', methods=['POST'])
def delete_model():
    
    form = request.form

    needed_session_fields = [USER_ID_FIELD_NAME, USERNAME_FIELD_NAME, ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME]
    needed_form_fields = [MODEL_NAME_FIELD_NAME]
    
    needed_fields_recieved = True

    for needed_session_field in needed_session_fields:
        if needed_session_field not in session:
            needed_fields_recieved = False
    
    for needed_form_field in needed_form_fields:
        if needed_form_field not in form:
            needed_fields_recieved = False
    
    if not needed_fields_recieved:
        return home()
    
    user_id = session[USER_ID_FIELD_NAME]
    username = session[USERNAME_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    env_name = session[ENV_NAME_FIELD_NAME]
    model_name = form[MODEL_NAME_FIELD_NAME]
    
    models = select_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                            ["*"], 
                            [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME], 
                            [user_id, env_id, model_name])
    
    if len(models) == 0:
        print("A model with this name doesn't exist!")
        return home()
    
    try:
        delete_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME], 
                       [user_id, env_id, model_name])
        
        path_to_env = USERS_DATA_FOLDER + username + "/" + ENVIRONMENTS_FOLDER_NAME + "/" + env_name + "/"
        path_to_model = path_to_env + "/" + MODELS_FOLDER_NAME + "/" + model_name + "/"
        
        shutil.rmtree(path_to_model)
    except:
        print("Couldn't delete the model!")
    
    return home()

@app.route('/predict', methods=['POST'])
def predict():

    form = request.form

    needed_session_fields = [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME]
    needed_form_fields = [MODEL_NAME_FIELD_NAME, SENTENCE_TO_PREDICT_FIELD_NAME]
    
    needed_fields_recieved = True

    for needed_session_field in needed_session_fields:
        if needed_session_field not in session:
            needed_fields_recieved = False
    
    for needed_form_field in needed_form_fields:
        if needed_form_field not in form:
            needed_fields_recieved = False
    
    if not needed_fields_recieved:
        return home()
    
    user_id = session[USER_ID_FIELD_NAME]
    envid = session[ENV_ID_FIELD_NAME]
    model_name = form[MODEL_NAME_FIELD_NAME]
    sent_to_predict = form[SENTENCE_TO_PREDICT_FIELD_NAME]

    model_tuples = select_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                                 [MODEL_PATH_FIELD_NAME, MODEL_TYPE_FIELD_NAME], 
                                 [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME], 
                                 [user_id, envid, model_name])
    
    if len(model_tuples) == 0:
        print("No model with this name found!")
        return home()
    
    path_to_model = model_tuples[0][0]

    try:
        new_model = NLPClassificationModel.load_model(path_to_model)
    except:
        print("Couldn't load the model!")
        return home()
    try:
        print(new_model.predict([sent_to_predict]))
    except:
        print("Something went wrong during the prediction...")
    
    return home()

@app.route('/create_dataset', methods=['POST'])
def create_dataset():
    pass

def initialize_server():

    path_to_users_data = "AlIve/UsersData/"

    if not os.path.exists(path_to_users_data):
        os.makedirs(path_to_users_data)
    
def reset_session():
    session[LOGGED_IN_FIELD_NAME] = False
    session.pop(USER_ID_FIELD_NAME)
    session.pop(USERNAME_FIELD_NAME)
    session.pop(USER_EMAIL_FIELD_NAME)
    session.pop(ENV_ID_FIELD_NAME)
    session.pop(ENV_NAME_FIELD_NAME)

def select_from_db(table_name:str, needed_fields:list=None, given_fields:list=None, given_values:list=None):

    if needed_fields == None:
        needed_fields = ["*"]
    
    if given_fields == None:
        given_fields = list()
    
    if given_values == None:
        given_values = list()
    
    if len(given_fields) != len(given_values):
        raise Exception("The number of fields given is different from the number of values!")
    
    query = "SELECT "

    for i, needed_field in enumerate(needed_fields):
        if i > 0:
            query += ", "
        query += needed_field
    
    query += " FROM " + table_name

    if len(given_fields) > 0:
        query += " WHERE "

        for i, given_field in enumerate(given_fields):
            if i > 0:
                query += " AND "
            
            query += (given_field + " = ")
            
            not_quoted_types = [int, float]
            
            field_value = None

            for not_quoted_type in not_quoted_types:
                if isinstance(given_values[i], not_quoted_type):
                    field_value = str(given_values[i])
                    break
            
            if field_value == None:
                field_value = "'{}'".format(given_values[i])
            
            query += field_value
    
    cursor = db_connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()

    return results

def insert_into_db(table_name:str, given_fields:list, given_values:list):

    if len(given_fields) != len(given_values):
        raise Exception("The number of fields given is different from the number of values!")
    
    query = "INSERT INTO " + table_name + " ("

    for i, given_field in enumerate(given_fields):
        if i > 0:
            query += ", "
        
        query += given_field
    
    query += ") VALUES ("

    for i, given_value in enumerate(given_values):
        if i > 0:
            query += ", "
        
        not_quoted_types = [int, float]
        
        field_value = None

        for not_quoted_type in not_quoted_types:
            if isinstance(given_value, not_quoted_type):
                field_value = str(given_value)
                break
        
        if field_value == None:
            field_value = "'{}'".format(given_values[i])
        
        query += field_value
    
    query += ")"

    cursor = db_connection.cursor()
    cursor.execute(query)
    db_connection.commit()
    cursor.close()

def delete_from_db(table_name:str, given_fields:list=None, given_values:list=None):

    if given_fields == None:
        given_fields = list()
    
    if given_values == None:
        given_values = list()
    
    if len(given_fields) != len(given_values):
        raise Exception("The number of fields given is different from the number of values!")
    
    query = "DELETE FROM " + table_name

    if len(given_fields) > 0:
        query += " WHERE "

        for i, given_field in enumerate(given_fields):
            if i > 0:
                query += " AND "
            
            query += (given_field + " = ")
            
            not_quoted_types = [int, float]
            
            field_value = None

            for not_quoted_type in not_quoted_types:
                if isinstance(given_values[i], not_quoted_type):
                    field_value = str(given_values[i])
                    break
            
            if field_value == None:
                field_value = "'{}'".format(given_values[i])
            
            query += field_value
    
    cursor = db_connection.cursor()
    cursor.execute(query)
    db_connection.commit()
    cursor.close()

if __name__ == "__main__":
    initialize_server()
    app.secret_key = os.urandom(12)
    app.run(debug=False,host='0.0.0.0', port=5000)
