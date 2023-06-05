import os
import re
from queue import Queue
from threading import Thread, Event

import mysql.connector as mysqlconn
from flask import Flask
from flask import Flask, flash, redirect, render_template, request, session, abort

import tensorflow as tf

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
QUEUE_INDEX_FIELD_NAME = "queue_index"

TRAINING_THREAD_INFO_FIELD_NAME = "training_thread"
IS_ALIVE_TRAINING_THREAD_FIELD_NAME = "is_alive"
WANT_TO_STOP_THREAD_FIELD_NAME = "want_to_stop"

ALIVE_DB_NAME = "alive_db"
ALIVE_DB_ADMIN_USERNAME = "GiuseppeVolpe"
ALIVE_DB_ADMIN_PASSWORD = "password"
ALIVE_DB_USERS_TABLE_NAME = "alive_users"
ALIVE_DB_ENVIRONMENTS_TABLE_NAME = "users_environments"
ALIVE_DB_MODELS_TABLE_NAME = "environments_models"
ALIVE_DB_DATASETS_TABLE_NAME = "environments_datasets"
ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME = "training_sessions"

STR_TYPE_NAME = "str"
INT_TYPE_NAME = "int"
BOOL_TYPE_NAME = "bool"

FINETUNABLE_FIELD_NAME = "finetunable"
BASEMODEL_FIELD_NAME = "base_model"
NUM_OF_CLASSES_FIELD_NAME = "num_of_classes"
ENCODER_TRAINABLE_FIELD_NAME = "encoder_trainable"
DROPOUT_RATE_FIELD_NAME = "dropout_rate"
OPTIMIZER_LR_FIELD_NAME = "optimizer_lr"
DATASET_CSV_FIELD_NAME = "dataset_csv"

TARGETS_FIELD_NAME = "targets"
NUM_OF_EPOCHS_FIELD_NAME = "num_of_epochs"
BATCH_SIZE_FIELD_NAME = "batch_size"
CHECKPOINT_PATH_FIELD_NAME = "checkpoint_path"
TARGETS_SEPARATOR = "|"

SLC_MODEL_TYPE = "SLCM"
TLC_MODEL_TYPE = "TLCM"

EMPTY_TARGET_VALUE = "None"

TEXT_FIELD_NAME = "text"
SENTENCE_IDX_FIELD_NAME = "sentence_idx"
WORD_FIELD_NAME = "word"
EXAMPLE_CATEGORY_FIELD_NAME = "example_category"

TEXT_COLUMN_NAME_FIELD_NAME = "text_column_name"
SENTENCE_IDX_COLUMN_NAME_FIELD_NAME = "sentence_idx_column_name"
WORD_COLUMN_NAME_FIELD_NAME = "word_column_name"

EXAMPLE_TRAIN_CATEGORY = "train"
EXAMPLE_VALIDATION_CATEGORY = "valid"
EXAMPLE_TEST_CATEGORY = "test"

SENTENCE_TO_PREDICT_FIELD_NAME = "sent"

ROOT_FOLDER = "AlIve"
USERS_DATA_FOLDER_NAME = "UsersData"
ENVIRONMENTS_FOLDER_NAME = "Environments"
MODELS_FOLDER_NAME = "Models"
DATASETS_FOLDER_NAME = "Datasets"
TRAINING_SESSIONS_FOLDER_NAME = "TrainingSessions"

USERS_DATA_FOLDER = ROOT_FOLDER + "/" + USERS_DATA_FOLDER_NAME + "/"

EXAMPLE_CATEGORIES = [EXAMPLE_TRAIN_CATEGORY, EXAMPLE_VALIDATION_CATEGORY, EXAMPLE_TEST_CATEGORY]
MODEL_TYPES = [SLC_MODEL_TYPE, TLC_MODEL_TYPE]

#endregion

app = Flask(__name__, template_folder='Templates')

app.config['UPLOAD_FOLDER'] = "UPLOAD_FOLDER"
app.config['MAX_CONTENT-PATH'] = 1000000

db_connection = mysqlconn.connect(user=ALIVE_DB_ADMIN_USERNAME, password=ALIVE_DB_ADMIN_PASSWORD, database=ALIVE_DB_NAME)

#region FORMS GETTERS

@app.route('/')
def home():
    if not session.get(LOGGED_IN_FIELD_NAME):
        return render_template('login.html')
    else:
        return render_template('index.html', 
                               available_models=get_available_models(), 
                               example_categories=EXAMPLE_CATEGORIES,
                               model_types=MODEL_TYPES)

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

#region USERS HANDLING

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

#endregion

#region ENVIRONMENTS HANDLING

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

#endregion

#region MODELS HANDLING

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
                            [MODEL_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME], 
                            [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME], 
                            [user_id, envid])
    
    max_id = 0

    for model in models:

        if model[0] > max_id:
            max_id = model[0]
        
        if model[1] == model_name:
            print("A model with this name already exists!")
            return home()
    
    new_model_id = max_id + 1

    path_to_env = USERS_DATA_FOLDER + username + "/" + ENVIRONMENTS_FOLDER_NAME + "/" + env_name + "/"
    path_to_model = path_to_env + "/" + MODELS_FOLDER_NAME + "/" + model_name + "/"
    
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)
    
    if model_type == SLC_MODEL_TYPE:
        new_model = SentenceLevelClassificationModel(model_name, finetunable)
        new_model.build(encoder_link, num_of_classes, preprocess_link, encoder_trainable, "pooled_output", 
                        dropout_rate=dropout_rate, optimizer_lr=optimizer_lr, 
                        additional_metrics=additional_metrics)
        new_model.save(path_to_model, True)
    elif model_type == TLC_MODEL_TYPE:
        new_model = TokenLevelClassificationModel(model_name, finetunable)
        new_model.build(preprocess_link, encoder_link, num_of_classes, encoder_trainable, 
                        dropout_rate=dropout_rate, optimizer_lr=optimizer_lr, additional_metrics=additional_metrics)
        new_model.save(path_to_model, True)
    
    insert_into_db(ALIVE_DB_MODELS_TABLE_NAME, 
                   [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME, 
                    MODEL_PATH_FIELD_NAME, MODEL_TYPE_FIELD_NAME, PUBLIC_FIELD_NAME], 
                   [user_id, envid, new_model_id, model_name, path_to_model, model_type, public])
    
    return home()

@app.route('/delete_model', methods=['POST'])
def delete_model():
    
    form = request.form

    needed_session_fields = [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME]
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
    env_id = session[ENV_ID_FIELD_NAME]
    model_name = form[MODEL_NAME_FIELD_NAME]
    
    models = select_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                            [MODEL_PATH_FIELD_NAME], 
                            [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME], 
                            [user_id, env_id, model_name])
    
    if len(models) == 0:
        print("A model with this name doesn't exist!")
        return home()
    
    model = models[0]
    path_to_model = model[0]
    
    try:
        delete_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME], 
                       [user_id, env_id, model_name])
        
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

#endregion

#region DATASET HANDLING

@app.route('/create_dataset', methods=['POST'])
def create_dataset():

    form = request.form
    needed_session_fields = [USER_ID_FIELD_NAME, USERNAME_FIELD_NAME, 
                             ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME]
    needed_form_fields = [DATASET_NAME_FIELD_NAME, DATASET_TYPE_FIELD_NAME]
    
    needed_fields_recieved = True

    for needed_session_field in needed_session_fields:
        if needed_session_field not in session:
            needed_fields_recieved = False
    
    for needed_form_field in needed_form_fields:
        if needed_form_field not in form:
            needed_fields_recieved = False
    
    if not needed_fields_recieved:
        print("Missing fields in the form!")
        return home()
    
    user_id = session[USER_ID_FIELD_NAME]
    username = session[USERNAME_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    env_name = session[ENV_NAME_FIELD_NAME]
    dataset_name = form[DATASET_NAME_FIELD_NAME]
    dataset_type = form[DATASET_TYPE_FIELD_NAME]
    public = PUBLIC_FIELD_NAME in form
    
    if len(dataset_name) <= 1:
        print("Invaild name!")
        return home()
    
    datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                              [DATASET_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                              [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME], 
                              [user_id, env_id])
    
    max_dataset_id = 0

    for dataset in datasets:
        
        if dataset[0] > max_dataset_id:
            max_dataset_id = dataset[0]

        if dataset[1] == dataset_name:
            print("A dataset with this name already exists!")
            return home()
    
    new_dataset_id = max_dataset_id + 1

    path_to_env = USERS_DATA_FOLDER + username + "/" + ENVIRONMENTS_FOLDER_NAME + "/" + env_name + "/"
    dataset_folder = path_to_env + "/" + DATASETS_FOLDER_NAME + "/"
    path_to_dataset = dataset_folder + dataset_name + ".pickle"

    if dataset_type == SLC_MODEL_TYPE:
        dataframe = pd.DataFrame({TEXT_FIELD_NAME:[], EXAMPLE_CATEGORY_FIELD_NAME:[]})
    elif dataset_type == TLC_MODEL_TYPE:
        dataframe = pd.DataFrame({SENTENCE_IDX_FIELD_NAME:[], WORD_FIELD_NAME:[], EXAMPLE_CATEGORY_FIELD_NAME:[]})
    else:
        print("Invalid dataset type!")
        return home()
    
    try:
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        
        dataframe.to_pickle(path_to_dataset)
    except:
        print("Couldn't create the dataset!")
        return home()
    
    try:
        insert_into_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_ID_FIELD_NAME, 
                        DATASET_NAME_FIELD_NAME, DATASET_PATH_FIELD_NAME, DATASET_TYPE_FIELD_NAME, 
                        PUBLIC_FIELD_NAME], 
                       [user_id, env_id, new_dataset_id, 
                        dataset_name, path_to_dataset, dataset_type, 
                        public])
    except Exception as ex:
        print("Couldn't insert dataset to database! " + str(ex))
    finally:
        return home()

@app.route('/delete_dataset', methods=['POST'])
def delete_dataset():
    
    form = request.form

    needed_session_fields = [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME]
    needed_form_fields = [DATASET_NAME_FIELD_NAME]
    
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
    env_id = session[ENV_ID_FIELD_NAME]
    dataset_name = form[DATASET_NAME_FIELD_NAME]
    
    datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                              [DATASET_PATH_FIELD_NAME], 
                              [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                              [user_id, env_id, dataset_name])
    
    if len(datasets) == 0:
        print("A dataset with this name doesn't exist!")
        return home()
    
    dataset = datasets[0]
    path_to_dataset = dataset[0]
    
    try:
        delete_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                       [user_id, env_id, dataset_name])
        
        os.remove(path_to_dataset)
    except:
        print("Couldn't delete the dataset!")
    
    return home()

@app.route('/import_csv_to_dataset', methods=['POST'])
def import_examples_to_dataset():
    
    form = request.form

    needed_session_fields = [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME]
    needed_form_fields = [DATASET_NAME_FIELD_NAME, EXAMPLE_CATEGORY_FIELD_NAME, 
                          TEXT_COLUMN_NAME_FIELD_NAME, SENTENCE_IDX_COLUMN_NAME_FIELD_NAME, 
                          WORD_COLUMN_NAME_FIELD_NAME]
    
    needed_fields_recieved = True

    for needed_session_field in needed_session_fields:
        if needed_session_field not in session:
            needed_fields_recieved = False
    
    for needed_form_field in needed_form_fields:
        if needed_form_field not in form:
            needed_fields_recieved = False
    
    if not needed_fields_recieved:
        print("Missing fields in the form!")
        return home()
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]

    dataset_name = form[DATASET_NAME_FIELD_NAME]
    category = form[EXAMPLE_CATEGORY_FIELD_NAME]
    text_column_name = form[TEXT_COLUMN_NAME_FIELD_NAME]
    sentence_idx_column_name = form[SENTENCE_IDX_COLUMN_NAME_FIELD_NAME]
    word_column_name = form[WORD_COLUMN_NAME_FIELD_NAME]
    
    datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                              [DATASET_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME, 
                               DATASET_TYPE_FIELD_NAME, DATASET_PATH_FIELD_NAME], 
                              [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                              [user_id, env_id, dataset_name])
    
    if len(datasets) == 0:
        print("A dataset with this name doesn't exist!")
        return home()
    
    existing_dataset = datasets[0]

    existing_dataset_id = existing_dataset[0]
    existing_dataset_type = existing_dataset[2]
    existing_dataset_path = existing_dataset[3]

    try:
        existing_dataset = pd.read_pickle(existing_dataset_path)
        
        infile = request.files[DATASET_CSV_FIELD_NAME]
        imported_dataset = pd.read_csv(infile, keep_default_na=False)
    except Exception as ex:
        print("Something went wrong when loading the dataset..." + str(ex))
        return home()
    
    needed_fields = []

    if text_column_name in imported_dataset.columns:
        imported_dataset.rename(columns={text_column_name: TEXT_FIELD_NAME})

    if sentence_idx_column_name in imported_dataset.columns:
        imported_dataset.rename(columns={sentence_idx_column_name: SENTENCE_IDX_FIELD_NAME})

    if word_column_name in imported_dataset.columns:
        imported_dataset.rename(columns={word_column_name: WORD_FIELD_NAME})

    if existing_dataset_type == SLC_MODEL_TYPE:
        needed_fields += [TEXT_FIELD_NAME]
    elif existing_dataset_type == TLC_MODEL_TYPE:
        needed_fields += [SENTENCE_IDX_FIELD_NAME, WORD_FIELD_NAME]

    for needed_field in needed_fields:
        if needed_field not in imported_dataset.columns:
            print("Trying to import from an invalid dataframe!")
            return home()

    for column in imported_dataset.columns:
        if column not in existing_dataset.columns:
            existing_dataset[column] = EMPTY_TARGET_VALUE

    for i, new_row in imported_dataset.iterrows():

        for column in existing_dataset:
            if column not in imported_dataset:
                new_row[column] = EMPTY_TARGET_VALUE
        
        new_row[EXAMPLE_CATEGORY_FIELD_NAME] = category

        if i % 10000 == 0:
            print(new_row)
        
        existing_dataset.loc[len(existing_dataset)] = new_row
    
    try:
        if os.path.exists(existing_dataset_path):
            os.remove(existing_dataset_path)
        
        existing_dataset.to_pickle(existing_dataset_path)
    except:
        print("Couldn't save the updated dataset!")
    
    print(existing_dataset)

    return home()

#endregion

#region TRAINING QUEUE HANDLING

@app.route('/add_to_train_queue', methods=['POST'])
def add_model_to_train_queue():

    form = request.form

    needed_session_fields = [USER_ID_FIELD_NAME, USERNAME_FIELD_NAME, 
                             ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME]
    needed_form_fields = [MODEL_NAME_FIELD_NAME, DATASET_NAME_FIELD_NAME, 
                          TARGETS_FIELD_NAME, NUM_OF_EPOCHS_FIELD_NAME, 
                          BATCH_SIZE_FIELD_NAME]
    
    needed_fields_recieved = True

    for needed_session_field in needed_session_fields:
        if needed_session_field not in session:
            needed_fields_recieved = False
    
    for needed_form_field in needed_form_fields:
        if needed_form_field not in form:
            needed_fields_recieved = False
    
    if not needed_fields_recieved:
        print("Missing fields!")
        return home()
    
    user_id = session[USER_ID_FIELD_NAME]
    username = session[USERNAME_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    env_name = session[ENV_NAME_FIELD_NAME]
    model_name = form[MODEL_NAME_FIELD_NAME]
    dataset_name = form[DATASET_NAME_FIELD_NAME]
    target = form[TARGETS_FIELD_NAME]
    num_of_epochs = form[NUM_OF_EPOCHS_FIELD_NAME]
    batch_size = form[BATCH_SIZE_FIELD_NAME]
    
    models = select_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                            [MODEL_ID_FIELD_NAME, MODEL_TYPE_FIELD_NAME, FINETUNABLE_FIELD_NAME], 
                            [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME], 
                            [user_id, env_id, model_name])
    
    if len(models) == 0:
        print("A model with this name doesn't exist!")
        return home()
    
    model_id = models[0][0]
    model_type = models[0][1]
    model_finetunable = models[0][2]
    
    datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                              [DATASET_ID_FIELD_NAME, DATASET_TYPE_FIELD_NAME], 
                              [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                              [user_id, env_id, dataset_name])
    
    if len(datasets) == 0:
        print("A dataset with this name doesn't exist!")
        return home()
    
    dataset_id = datasets[0][0]
    dataset_type = datasets[0][1]

    if model_type != dataset_type:
        print("Can't add to train queue, dataset type is invalid!")
        return home()
    
    queue_in_this_env = select_from_db(ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME, 
                                       [QUEUE_INDEX_FIELD_NAME, MODEL_ID_FIELD_NAME], 
                                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME], 
                                       [user_id, env_id])
    
    max_id = 0

    for training_session in queue_in_this_env:
        
        if training_session[0] > max_id:
            max_id = training_session[0]
        
        if model_id == training_session[1] and not model_finetunable:
            print("Can't add this model to the queue, is already trained and not finetunable!")
            return home()
        
    new_id = max_id + 1

    path_to_env = USERS_DATA_FOLDER + username + "/" + ENVIRONMENTS_FOLDER_NAME + "/" + env_name + "/"
    path_to_training_sessions = path_to_env + "/" + TRAINING_SESSIONS_FOLDER_NAME + "/"
    checkpoint_name = str(model_id) + str(dataset_id) + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    checkpoint_path = path_to_training_sessions + "/" + checkpoint_name + "/"
    
    try:
        insert_into_db(ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME,
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, QUEUE_INDEX_FIELD_NAME, 
                        MODEL_ID_FIELD_NAME, DATASET_ID_FIELD_NAME, TARGETS_FIELD_NAME, 
                        NUM_OF_EPOCHS_FIELD_NAME, BATCH_SIZE_FIELD_NAME, CHECKPOINT_PATH_FIELD_NAME], 
                        [user_id, env_id, new_id, 
                         model_id, dataset_id, target,
                         num_of_epochs, batch_size, checkpoint_path])
    except Exception as ex:
        print("Couldn't add this model to train queue! " + str(ex))
    
    return home()

@app.route('/remove_from_train_queue', methods=['POST'])
def remove_session_from_train_queue():

    form = request.form

    needed_session_fields = [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME]
    needed_form_fields = [QUEUE_INDEX_FIELD_NAME]
    
    needed_fields_recieved = True

    for needed_session_field in needed_session_fields:
        if needed_session_field not in session:
            needed_fields_recieved = False
    
    for needed_form_field in needed_form_fields:
        if needed_form_field not in form:
            needed_fields_recieved = False
    
    if not needed_fields_recieved:
        print("Missing fields!")
        return home()
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    queue_index = form[QUEUE_INDEX_FIELD_NAME]

    try:
        training_sessions = select_from_db(ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME, 
                                           [CHECKPOINT_PATH_FIELD_NAME], 
                                           [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, QUEUE_INDEX_FIELD_NAME], 
                                           [user_id, env_id, queue_index])
        
        if len(training_sessions) == 0:
            print("There is no training session at this index!")
            return home()
        
        training_session = training_sessions[0]
        checkpoint_path = training_session[0]

        delete_from_db(ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, QUEUE_INDEX_FIELD_NAME], 
                       [user_id, env_id, queue_index])
        
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        
    except:
        print("Couldn't delete training session!")
    
    return home()

@app.route('/start_train', methods=['POST'])
def start_train():

    if TRAINING_THREAD_INFO_FIELD_NAME in session:
        if session[TRAINING_THREAD_INFO_FIELD_NAME][IS_ALIVE_TRAINING_THREAD_FIELD_NAME]:
            print("The training has already started!")
            return home()

    needed_session_fields = [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME]

    for needed_session_field in needed_session_fields:
        if needed_session_field not in session:
            return home()
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    
    training_thread_info = {IS_ALIVE_TRAINING_THREAD_FIELD_NAME : False, 
                            WANT_TO_STOP_THREAD_FIELD_NAME : False}
    session[TRAINING_THREAD_INFO_FIELD_NAME] = training_thread_info

    lambda_training_function = lambda : train_queue(user_id, env_id, training_thread_info)
    training_thread = Thread(target=lambda_training_function, daemon=True, name='Monitor')
    training_thread.start()

    return home()

def train_queue(user_id:int, env_id:int, training_thread_info:dict):
    
    queue_in_this_env = select_from_db(ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME, 
                                       [QUEUE_INDEX_FIELD_NAME, 
                                        MODEL_ID_FIELD_NAME, DATASET_ID_FIELD_NAME, 
                                        TARGETS_FIELD_NAME, NUM_OF_EPOCHS_FIELD_NAME,
                                        BATCH_SIZE_FIELD_NAME, CHECKPOINT_PATH_FIELD_NAME], 
                                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME], 
                                       [user_id, env_id])
    
    if len(queue_in_this_env) == 0:
        print("Nothing on train queue!")
        return
    
    queue_in_this_env.sort(key=(lambda session:session[0]))

    training_sessions = Queue()

    for training_session in queue_in_this_env:
        training_sessions.put(training_session)
    
    training_thread_info[IS_ALIVE_TRAINING_THREAD_FIELD_NAME] = True

    while training_sessions.empty:
        
        if training_thread_info[WANT_TO_STOP_THREAD_FIELD_NAME]:
            training_thread_info[IS_ALIVE_TRAINING_THREAD_FIELD_NAME] = False
            print('The training thread was stopped prematurely.')
            break
        
        try:
            training_session = training_sessions.get()

            current_queue_index = training_session[0]
            model_id = training_session[1]
            dataset_id = training_session[2]
            targets = training_session[3].split(TARGETS_SEPARATOR)
            epochs_left = training_session[4]
            batch_size = training_session[5]
            checkpoint_path = training_session[6]

            models = select_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                                    [MODEL_PATH_FIELD_NAME, MODEL_TYPE_FIELD_NAME], 
                                    [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_ID_FIELD_NAME], 
                                    [user_id, env_id, model_id])
            
            if len(models) == 0:
                print("Couldn't find the model specified in this train session!")
                continue
            
            model = models[0]

            path_to_model = model[0]
            model_type = model[1]
            
            datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                                      [DATASET_PATH_FIELD_NAME], 
                                      [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_ID_FIELD_NAME], 
                                      [user_id, env_id, dataset_id])
            
            if len(datasets) == 0:
                print("Couldn't find the model specified in this train session!")
                continue
            
            dataset = datasets[0]

            path_to_dataset = dataset[0]

            loaded_model = NLPClassificationModel.load_model(path_to_model)
            loaded_dataset = pd.read_pickle(path_to_dataset)
            
            train = loaded_dataset[loaded_dataset[EXAMPLE_CATEGORY_FIELD_NAME] == EXAMPLE_TRAIN_CATEGORY]
            valid = loaded_dataset[loaded_dataset[EXAMPLE_CATEGORY_FIELD_NAME] == EXAMPLE_VALIDATION_CATEGORY]
            test = loaded_dataset[loaded_dataset[EXAMPLE_CATEGORY_FIELD_NAME] == EXAMPLE_TEST_CATEGORY]
            
            if model_type == SLC_MODEL_TYPE:
                data = SentenceLevelClassificationData(train, valid, test, TEXT_FIELD_NAME, targets[0])
            elif model_type == TLC_MODEL_TYPE:
                data = TokenLevelClassificationData(train, valid, test, loaded_model.tokenize, 
                                                    SENTENCE_IDX_FIELD_NAME, WORD_FIELD_NAME, 
                                                    targets[0])
            
            epochs_updating_callback = UpdateDBCallback(user_id, env_id, 
                                                        current_queue_index, db_connection)
            additional_callbacks = [epochs_updating_callback]
            
            loaded_model.train(data, epochs_left, batch_size, checkpoint_path, additional_callbacks)
            loaded_model.save(path_to_model, True)
            shutil.rmtree(checkpoint_path)

            delete_from_db(ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME, 
                        [ENV_ID_FIELD_NAME, QUEUE_INDEX_FIELD_NAME], 
                        [env_id, current_queue_index])
        except Exception as ex:
            print(ex)
            continue
    
    training_thread_info[IS_ALIVE_TRAINING_THREAD_FIELD_NAME] = False

@app.route('/stop_train', methods=['POST'])
def stop_train():

    needed_session_fields = [TRAINING_THREAD_INFO_FIELD_NAME]

    for needed_session_field in needed_session_fields:
        if needed_session_field not in session:
            return home()
    
    training_thread_info = session[TRAINING_THREAD_INFO_FIELD_NAME]
    training_thread_info[WANT_TO_STOP_THREAD_FIELD_NAME] = True

#endregion

def initialize_server():

    path_to_users_data = USERS_DATA_FOLDER

    if not os.path.exists(path_to_users_data):
        os.makedirs(path_to_users_data)
    
def reset_session():
    session[LOGGED_IN_FIELD_NAME] = False
    session.pop(USER_ID_FIELD_NAME)
    session.pop(USERNAME_FIELD_NAME)
    session.pop(USER_EMAIL_FIELD_NAME)
    session.pop(ENV_ID_FIELD_NAME)
    session.pop(ENV_NAME_FIELD_NAME)
    session.pop(TRAINING_THREAD_INFO_FIELD_NAME)

#region DB UTILITIES

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

def update_db(table_name:str, 
              fields_to_update:list=None, updated_values:list=None,
              given_fields:list=None, given_values:list=None):

    if fields_to_update == None:
        fields_to_update = list()
    
    if updated_values == None:
        updated_values = list()
    
    if len(fields_to_update) == 0:
        raise Exception("Nothing to update!")

    if given_fields == None:
        given_fields = list()
    
    if given_values == None:
        given_values = list()
    
    if len(fields_to_update) != len(updated_values):
        raise Exception("The number of fields to update is different from the number of values!")
    
    if len(given_fields) != len(given_values):
        raise Exception("The number of fields given is different from the number of values!")
    
    query = "UPDATE " + table_name + " SET "

    for i, field_to_update in enumerate(fields_to_update):

        updated_value = updated_values[i]

        not_quoted_types = [int, float]
        
        if updated_value in not_quoted_type:
            updated_value = str(updated_value)
        else:
            updated_value = "'{}'".format(updated_value)

        if i != 0:
            query += ", "

        query += field_to_update + " = " + updated_value

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

def execute_custom_update_query(query):
    cursor = db_connection.cursor()
    cursor.execute(query)
    db_connection.commit()
    cursor.close()

#endregion

class UpdateDBCallback(tf.keras.callbacks.Callback):

    def __init__(self, user_id:int, env_id:int, current_queue_index:int, db_connection):
        super().__init__()
        self.__user_id = user_id
        self.__env_id = env_id
        self.__current_queue_index = current_queue_index
    
    def on_epoch_end(self, epoch, logs=None):

        user_id = self.__user_id
        env_id = self.__env_id
        current_queue_index = self.__current_queue_index

        update_epochs_left_query = "UPDATE " + ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME + " SET "
        update_epochs_left_query += NUM_OF_EPOCHS_FIELD_NAME + " = " + NUM_OF_EPOCHS_FIELD_NAME + " - 1 "
        update_epochs_left_query += " WHERE " + USER_ID_FIELD_NAME + " = " + str(user_id) + " AND "
        update_epochs_left_query += ENV_ID_FIELD_NAME + " = " + str(env_id) + " AND "
        update_epochs_left_query += QUEUE_INDEX_FIELD_NAME + " = " + str(current_queue_index)
        
        cursor = db_connection.cursor()
        cursor.execute(update_epochs_left_query)
        db_connection.commit()
        cursor.close()

if __name__ == "__main__":
    initialize_server()
    app.secret_key = os.urandom(12)
    app.run(debug=False,host='0.0.0.0', port=5000)
