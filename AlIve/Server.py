import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import re
from queue import Queue
from threading import Thread

import mysql.connector as mysqlconn
from flask import Flask, jsonify, request
from flask_cors import CORS

import tensorflow as tf

from ModelsAndDatasets import *

#region CONSTS

MIN_USERNAME_LENGTH = 2
MIN_PASSWORD_LENGTH = 8

USER_ID_FIELD_NAME = "user_id"
USERNAME_FIELD_NAME = "username"
USER_PASSWORD_FIELD_NAME = "user_password"
USER_EMAIL_FIELD_NAME = "user_email"
ENV_ID_FIELD_NAME = "env_id"
ENV_NAME_FIELD_NAME = "env_name"
ENV_PATH_FIELD_NAME = "env_path"
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
OUTPUT_SHAPE_FIELD_NAME = "output_shape"
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
TRAINING_GRAPHS_FOLDER_NAME = "TrainingGraphs"

USERS_DATA_FOLDER = ROOT_FOLDER + "/" + USERS_DATA_FOLDER_NAME + "/"

EXAMPLE_CATEGORIES = [EXAMPLE_TRAIN_CATEGORY, EXAMPLE_VALIDATION_CATEGORY, EXAMPLE_TEST_CATEGORY]
MODEL_TYPES = [SLC_MODEL_TYPE, TLC_MODEL_TYPE]

SESSION_FIELD_NAME = "session"

#endregion

app = Flask(__name__, template_folder='Templates')
app.config['UPLOAD_FOLDER'] = "UPLOAD_FOLDER"
app.config['MAX_CONTENT-PATH'] = 1000000
CORS(app)

DB_CONNECTION = mysqlconn.connect(user=ALIVE_DB_ADMIN_USERNAME, password=ALIVE_DB_ADMIN_PASSWORD, database=ALIVE_DB_NAME)

LOADED_MODELS = dict()
TRAIN_QUEUES = dict()
TRAIN_QUEUES_PROGRESS_INFOS = dict()

#region USERS HANDLING

@app.route('/signup', methods=['POST'])
def signup():

    json = request.json

    needed_fields = [USERNAME_FIELD_NAME, USER_PASSWORD_FIELD_NAME, USER_EMAIL_FIELD_NAME]

    for needed_field in needed_fields:
        if needed_field not in json:
            return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    username = json[USERNAME_FIELD_NAME]
    user_email = json[USER_EMAIL_FIELD_NAME]
    user_password = json[USER_PASSWORD_FIELD_NAME]

    errors = list()

    if len(username) < MIN_USERNAME_LENGTH:
        errors.append("The length of the username should be at least {}!".format(MIN_USERNAME_LENGTH))
    
    email_regex = "^[a-zA-Z0-9][a-zA-Z0-9.!#$%&'*+-/=?^_`{|}~]*?[a-zA-Z0-9._-]?@[a-zA-Z0-9][a-zA-Z0-9._-]*?[a-zA-Z0-9]?\\.[a-zA-Z]{2,63}$"
    
    if not bool( re.match(email_regex, user_email) ):
        errors.append("This is not a valid email!")
    
    if len(user_password) < MIN_PASSWORD_LENGTH:
        errors.append("The length of the password should be at least {}!".format(MIN_PASSWORD_LENGTH))
    
    if len(errors) > 0:
        return compose_response("Errors in the form compilation!", errors, RETURNING_ERRORS_CODE)
    
    usernames = select_from_db(ALIVE_DB_USERS_TABLE_NAME, 
                               [USERNAME_FIELD_NAME], 
                               [USERNAME_FIELD_NAME], 
                               [username])
    
    if len(usernames) > 0:
        errors.append("This username is already taken!")
    
    email_addresses = select_from_db(ALIVE_DB_USERS_TABLE_NAME, 
                                     [USER_EMAIL_FIELD_NAME], 
                                     [USER_EMAIL_FIELD_NAME], 
                                     [user_email])
    
    if len(email_addresses) > 0:
        errors.append("This email address is already taken!")
    
    if len(errors) > 0:
        return compose_response("Errors found!", errors, RETURNING_ERRORS_CODE)
    
    try:
        insert_into_db(ALIVE_DB_USERS_TABLE_NAME, 
                       [USERNAME_FIELD_NAME, USER_PASSWORD_FIELD_NAME, USER_EMAIL_FIELD_NAME],
                       [username, user_password, user_email])
    except Exception as ex:
        print(ex)
        return compose_response("Couldn't add user!", code=FAILURE_CODE)
    
    user_space_path = USERS_DATA_FOLDER + username + "/"
    user_environments_path = user_space_path + "/" + ENVIRONMENTS_FOLDER_NAME + "/"

    if not os.path.exists(user_environments_path):
        os.makedirs(user_environments_path)
    
    return compose_response("User added succesfully!")

@app.route('/login', methods=['POST'])
def login():

    json = request.json

    needed_fields = [USERNAME_FIELD_NAME, USER_PASSWORD_FIELD_NAME]

    for needed_field in needed_fields:
        if needed_field not in json:
            return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    username = json[USERNAME_FIELD_NAME]
    inserted_password = json[USER_PASSWORD_FIELD_NAME]

    users = select_from_db(ALIVE_DB_USERS_TABLE_NAME, 
                           ["*"], 
                           [USERNAME_FIELD_NAME], 
                           [username])
    
    if len(users) == 0:
        return compose_response("This user doesn't exist!", code=FAILURE_CODE)
    
    user_tuple = users[0]
    
    user_id = user_tuple[0]
    correct_password = user_tuple[2]
    user_email = user_tuple[3]

    if inserted_password != correct_password:
        return compose_response("Wrong password!", code=FAILURE_CODE)
    
    data = {
        USER_ID_FIELD_NAME : user_id, 
        USERNAME_FIELD_NAME : username, 
        USER_EMAIL_FIELD_NAME : user_email
    }
    
    return compose_response("Login done succesfully!", data)

@app.route('/logout')
def logout():
    return compose_response("Logout done succesfully!")

#endregion

#region ENVIRONMENTS HANDLING

@app.route('/create_env', methods=['POST'])
def create_environment():
    
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json

    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_NAME_FIELD_NAME not in json:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    username = session[USERNAME_FIELD_NAME]
    env_name = json[ENV_NAME_FIELD_NAME]
    public = PUBLIC_FIELD_NAME in json
    
    if len(env_name) < 2:
        return compose_response("The length of the name should be at least 2!", code=FAILURE_CODE)
    
    try:
        environments = select_from_db(ALIVE_DB_ENVIRONMENTS_TABLE_NAME, 
                                      [ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                                      [USER_ID_FIELD_NAME], 
                                      [user_id])
        
        max_env_id = 0

        for environment in environments:
            
            if environment[0] > max_env_id:
                max_env_id = environment[0]

            if environment[1] == env_name:
                return compose_response("An environment with this name already exists!", code=FAILURE_CODE)
        
        new_env_id = max_env_id + 1

        path_to_env = USERS_DATA_FOLDER + username + "/" + ENVIRONMENTS_FOLDER_NAME + "/" + env_name + "/"

        if os.path.exists(path_to_env):
            shutil.rmtree(path_to_env)
        
        os.makedirs(path_to_env)

        insert_into_db(ALIVE_DB_ENVIRONMENTS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME, 
                        ENV_PATH_FIELD_NAME, PUBLIC_FIELD_NAME], 
                       [user_id, new_env_id, env_name, path_to_env, public])
    except:
        return compose_response("Something went wrong, couldn't create the environment...", code=FAILURE_CODE)
    
    return compose_response("Environment created succesfully!")

@app.route('/delete_env', methods=['POST'])
def delete_environment():
    
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json

    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_NAME_FIELD_NAME not in json:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_name = json[ENV_NAME_FIELD_NAME]
    
    try:
        environments = select_from_db(ALIVE_DB_ENVIRONMENTS_TABLE_NAME, 
                                      [ENV_PATH_FIELD_NAME], 
                                      [USER_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                                      [user_id, env_name])
        
        if len(environments) == 0:
            return compose_response("An environment with this name doesn't exist!", code=FAILURE_CODE)
        
        path_to_env = environments[0][0]
        shutil.rmtree(path_to_env)

        delete_from_db(ALIVE_DB_ENVIRONMENTS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                       [user_id, env_name])
    except:
        return compose_response("Something went wrong, couldn't delete the environment...", code=FAILURE_CODE)
    
    return compose_response("Environment deleted succesfully!")

@app.route('/select_env', methods=['POST'])
def select_environment():
    
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json

    env_id = None
    env_name = None
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    
    if ENV_ID_FIELD_NAME in json:
        env_id = json[ENV_ID_FIELD_NAME]
    
    if ENV_NAME_FIELD_NAME in json:
        env_name = json[ENV_NAME_FIELD_NAME]
    
    if env_id == None and env_name == None:
        return compose_response("No environment given!", code=MISSING_FIELDS_CODE)
    
    try:
        if env_id != None:
            environments = select_from_db(ALIVE_DB_ENVIRONMENTS_TABLE_NAME, 
                                          [ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                                          [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME], 
                                          [user_id, env_id])
        elif env_name != None:
            environments = select_from_db(ALIVE_DB_ENVIRONMENTS_TABLE_NAME, 
                                          [ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                                          [USER_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                                          [user_id, env_name])
        
        if len(environments) == 0:
            return compose_response("Inexisting environment!", code=FAILURE_CODE)
        
    except:
        return compose_response("Something went wrong, couldn't select the environment...", code=FAILURE_CODE)
    
    data = {
        ENV_ID_FIELD_NAME : environments[0][0],
        ENV_NAME_FIELD_NAME : environments[0][1]
    }
    
    return compose_response("Environment selected successfully!", data)

@app.route('/get_user_envs', methods=['POST'])
def get_user_envs():

    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    
    environments = select_from_db(ALIVE_DB_ENVIRONMENTS_TABLE_NAME, 
                                  [ENV_ID_FIELD_NAME, ENV_NAME_FIELD_NAME], 
                                  [USER_ID_FIELD_NAME], 
                                  [user_id])
    
    data = [{"value" : {"id" : env[0], "name" : env[1]}, "text" : env[1]} for env in environments]

    return compose_response("Environments fetched!", data)

#endregion

#region MODELS HANDLING

@app.route('/create_model', methods=['POST'])
def create_model():
    
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    needed_form_fields = [MODEL_NAME_FIELD_NAME, MODEL_TYPE_FIELD_NAME, 
                          BASEMODEL_FIELD_NAME, OUTPUT_SHAPE_FIELD_NAME, 
                          DROPOUT_RATE_FIELD_NAME, OPTIMIZER_LR_FIELD_NAME]
    
    for needed_form_field in needed_form_fields:
        if needed_form_field not in json:
            return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    username = session[USERNAME_FIELD_NAME]
    envid = session[ENV_ID_FIELD_NAME]
    env_name = session[ENV_NAME_FIELD_NAME]
    model_name = json[MODEL_NAME_FIELD_NAME]
    model_type = json[MODEL_TYPE_FIELD_NAME]
    finetunable = FINETUNABLE_FIELD_NAME in json
    base_model = json[BASEMODEL_FIELD_NAME]
    output_shape = int(json[OUTPUT_SHAPE_FIELD_NAME])
    encoder_trainable = ENCODER_TRAINABLE_FIELD_NAME in json
    dropout_rate = float(json[DROPOUT_RATE_FIELD_NAME])
    optimizer_lr = float(json[OPTIMIZER_LR_FIELD_NAME])
    additional_metrics = []
    public = PUBLIC_FIELD_NAME in json
    
    try:
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
                return compose_response("A model with this name already exists in this environment!", code=FAILURE_CODE)
        
        new_model_id = max_id + 1

        path_to_env = USERS_DATA_FOLDER + username + "/" + ENVIRONMENTS_FOLDER_NAME + "/" + env_name + "/"
        path_to_model = path_to_env + MODELS_FOLDER_NAME + "/" + model_name + "/"
        
        if os.path.exists(path_to_model):
            shutil.rmtree(path_to_model)
        
        os.makedirs(path_to_model)
        
        if model_type == SLC_MODEL_TYPE:
            new_model = SentenceLevelClassificationModel(model_name, finetunable)
            new_model.build(encoder_link, output_shape, preprocess_link, encoder_trainable, "pooled_output", 
                            dropout_rate=dropout_rate, optimizer_lr=optimizer_lr, 
                            additional_metrics=additional_metrics)
            new_model.save(path_to_model, True)
        elif model_type == TLC_MODEL_TYPE:
            new_model = TokenLevelClassificationModel(model_name, finetunable)
            new_model.build(preprocess_link, encoder_link, output_shape, encoder_trainable, 
                            dropout_rate=dropout_rate, optimizer_lr=optimizer_lr, additional_metrics=additional_metrics)
            new_model.save(path_to_model, True)
        
        insert_into_db(ALIVE_DB_MODELS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME, 
                        MODEL_PATH_FIELD_NAME, MODEL_TYPE_FIELD_NAME, PUBLIC_FIELD_NAME], 
                        [user_id, envid, new_model_id, model_name, path_to_model, model_type, public])
    except Exception as ex:
        print(ex)
        return compose_response("Something went wrong, couldn't create the model...", code=FAILURE_CODE)
    
    return compose_response("Model created succesfully!")

@app.route('/delete_model', methods=['POST'])
def delete_model():
    
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    if MODEL_NAME_FIELD_NAME not in json:
        return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    model_name = json[MODEL_NAME_FIELD_NAME]
    
    key = "{}_{}".format(user_id, env_id)

    if key in TRAIN_QUEUES.keys():
        if TRAIN_QUEUES[key].is_alive():
            return compose_response("This function is disabled when training is in progress!", code=FAILURE_CODE)
    
    try:
        models = select_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                                [MODEL_PATH_FIELD_NAME], 
                                [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME], 
                                [user_id, env_id, model_name])
        
        if len(models) == 0:
            return compose_response("A model with this name doesn't exist!", code=FAILURE_CODE)
        
        path_to_model = models[0][0]
        
        delete_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME], 
                       [user_id, env_id, model_name])
        
        if os.path.exists(path_to_model) and os.path.isdir(path_to_model):
            shutil.rmtree(path_to_model)
    except:
        return compose_response("Something went wrong, couldn't delete the model...", code=FAILURE_CODE)
    
    return compose_response("Model deleted succesfully!")

@app.route('/predict', methods=['POST'])
def predict():
    
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    needed_form_fields = [MODEL_NAME_FIELD_NAME, SENTENCE_TO_PREDICT_FIELD_NAME]
    
    for needed_form_field in needed_form_fields:
        if needed_form_field not in json:
            return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    model_name = json[MODEL_NAME_FIELD_NAME]
    sent_to_predict = json[SENTENCE_TO_PREDICT_FIELD_NAME]
    
    key = "{}_{}".format(user_id, env_id)

    if key in TRAIN_QUEUES.keys():
        if TRAIN_QUEUES[key].is_alive():
            return compose_response("This function is disabled when training is in progress!", code=FAILURE_CODE)
    
    try:
        models = select_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                                [MODEL_PATH_FIELD_NAME, MODEL_TYPE_FIELD_NAME], 
                                [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, 
                                 MODEL_NAME_FIELD_NAME], 
                                 [user_id, env_id, model_name])
        
        if len(models) == 0:
            return compose_response("No model with this name found!", code=FAILURE_CODE)
        
        model = models[0]
        
        path_to_model = model[0]
        model_type = model[1]
        
        key = "{}_{}_{}".format(user_id, env_id, model_name)

        if key not in LOADED_MODELS.keys():
            LOADED_MODELS[key] = NLPClassificationModel.load_model(path_to_model)
        
        print("Loaded model for prediction!")
        
        new_model = LOADED_MODELS[key]

        data = {"prediction" : ""}
        
        if model_type == SLC_MODEL_TYPE:
            prediction, prob = new_model.predict(sent_to_predict)
            
            data = {
                "prediction" : str(prediction),
                "prob" : prob,
            }
        elif model_type == TLC_MODEL_TYPE:
            prediction, entities = new_model.predict(sent_to_predict) 
            
            data = {
                "prediction" : str(prediction),
                "entities" : entities,
            }

        return compose_response("Prediction done successfully!", data)
    except Exception as ex:
        print(ex)
        return compose_response("Something went wrong during the prediction...", code=FAILURE_CODE)

@app.route('/get_env_models', methods=['POST'])
def get_env_models():

    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)
    
    session = request.json[SESSION_FIELD_NAME]
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("No environment selected!", code=FAILURE_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    
    models = select_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                            [MODEL_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME, MODEL_TYPE_FIELD_NAME], 
                            [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME],
                            [user_id, env_id])
    
    data = [{"value" : {"id" : model[0], "name" : model[1], "type": model[2]}, "text" : model[1]} for model in models]

    return compose_response("Models fetched!", data)

@app.route('/get_model_training_graphs', methods=['POST'])
def get_model_training_graphs():
    
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)
    
    session = request.json[SESSION_FIELD_NAME]

    json = request.json
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("No environment selected!", code=FAILURE_CODE)
    
    if MODEL_NAME_FIELD_NAME not in json:
        return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    model_name = json[MODEL_NAME_FIELD_NAME]
    
    try:
        models = select_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                                [MODEL_PATH_FIELD_NAME], 
                                [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME], 
                                [user_id, env_id, model_name])
        
        if len(models) == 0:
            return compose_response("A model with this name doesn't exist!", code=FAILURE_CODE)
        
        path_to_model = models[0][0]
        path_to_graphs = path_to_model + TRAINING_GRAPHS_FOLDER_NAME + "/"

        graph_images = []

        for image_name in os.listdir(path_to_graphs):
            image_path = path_to_graphs + image_name
            if os.path.isfile(image_path):
                import base64
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                    graph_images.append(encoded_image)
        
        return compose_response("Fetched training graphs succesfully!", data=graph_images)
    except Exception as ex:
        print(ex)
        return compose_response("Couldn't fetch the training graphs...", code=FAILURE_CODE)

#endregion

#region DATASET HANDLING

@app.route('/create_dataset', methods=['POST'])
def create_dataset():
    
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    needed_form_fields = [DATASET_NAME_FIELD_NAME, DATASET_TYPE_FIELD_NAME]
    
    for needed_form_field in needed_form_fields:
        if needed_form_field not in json:
            return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    username = session[USERNAME_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    env_name = session[ENV_NAME_FIELD_NAME]
    dataset_name = json[DATASET_NAME_FIELD_NAME]
    dataset_type = json[DATASET_TYPE_FIELD_NAME]
    public = PUBLIC_FIELD_NAME in json
    
    if len(dataset_name) < 2:
        return compose_response("The length of the name should be at least 2!", code=FAILURE_CODE)
    
    try:
        datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                                  [DATASET_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                                  [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME], 
                                  [user_id, env_id])
        
        max_dataset_id = 0

        for dataset in datasets:
            
            if dataset[0] > max_dataset_id:
                max_dataset_id = dataset[0]

            if dataset[1] == dataset_name:
                return compose_response("A dataset with this name already exists in this environment!", code=FAILURE_CODE)
        
        new_dataset_id = max_dataset_id + 1

        path_to_env = USERS_DATA_FOLDER + username + "/" + ENVIRONMENTS_FOLDER_NAME + "/" + env_name + "/"
        dataset_folder = path_to_env + DATASETS_FOLDER_NAME + "/"
        path_to_dataset = dataset_folder + dataset_name + ".pickle"

        if dataset_type == SLC_MODEL_TYPE:
            dataframe = pd.DataFrame({TEXT_FIELD_NAME:[], EXAMPLE_CATEGORY_FIELD_NAME:[]})
        elif dataset_type == TLC_MODEL_TYPE:
            dataframe = pd.DataFrame({SENTENCE_IDX_FIELD_NAME:[], WORD_FIELD_NAME:[], EXAMPLE_CATEGORY_FIELD_NAME:[]})
        else:
            return compose_response("Invalid dataset type!", code=FAILURE_CODE)
        
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        
        if os.path.exists(path_to_dataset):
            os.remove(path_to_dataset)
        
        dataframe.to_pickle(path_to_dataset)
        
        insert_into_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_ID_FIELD_NAME, 
                        DATASET_NAME_FIELD_NAME, DATASET_PATH_FIELD_NAME, DATASET_TYPE_FIELD_NAME, 
                        PUBLIC_FIELD_NAME], 
                        [user_id, env_id, new_dataset_id, 
                         dataset_name, path_to_dataset, dataset_type, 
                         public])
    except:
        return compose_response("Something went wrong, couldn't create the dataset...", code=FAILURE_CODE)
    
    return compose_response("Dataset created succesfully!")

@app.route('/delete_dataset', methods=['POST'])
def delete_dataset():
    
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    if DATASET_NAME_FIELD_NAME not in json:
        return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    dataset_name = json[DATASET_NAME_FIELD_NAME]
    
    key = "{}_{}".format(user_id, env_id)

    if key in TRAIN_QUEUES.keys():
        if TRAIN_QUEUES[key].is_alive():
            return compose_response("This function is disabled when training is in progress!", code=FAILURE_CODE)
    
    try:
        datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                                  [DATASET_PATH_FIELD_NAME], 
                                  [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                                  [user_id, env_id, dataset_name])
        
        if len(datasets) == 0:
            return compose_response("A dataset with this name doesn't exist!", code=FAILURE_CODE)
        
        path_to_dataset = datasets[0][0]
        
        delete_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                       [user_id, env_id, dataset_name])
        
        if os.path.exists(path_to_dataset):
            os.remove(path_to_dataset)
    except:
        return compose_response("Something went wrong, couldn't delete the dataset...", code=FAILURE_CODE)
    
    return compose_response("Dataset deleted succesfully!")

@app.route('/import_csv_to_dataset', methods=['POST'])
def import_examples_to_dataset():

    #region Composing session

    if USER_ID_FIELD_NAME not in request.form or ENV_ID_FIELD_NAME not in request.form:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = {
        USER_ID_FIELD_NAME : request.form[USER_ID_FIELD_NAME], 
        ENV_ID_FIELD_NAME : request.form[ENV_ID_FIELD_NAME]
    }

    #endregion
    
    form = request.form
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    needed_form_fields = [DATASET_NAME_FIELD_NAME, EXAMPLE_CATEGORY_FIELD_NAME, 
                          TEXT_COLUMN_NAME_FIELD_NAME, SENTENCE_IDX_COLUMN_NAME_FIELD_NAME, 
                          WORD_COLUMN_NAME_FIELD_NAME]
    
    for needed_form_field in needed_form_fields:
        if needed_form_field not in form:
            return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    if DATASET_CSV_FIELD_NAME not in request.files:
        return compose_response("Didn't recieve needed file!", code=MISSING_FIELDS_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]

    dataset_name = form[DATASET_NAME_FIELD_NAME]
    category = form[EXAMPLE_CATEGORY_FIELD_NAME]
    text_column_name = form[TEXT_COLUMN_NAME_FIELD_NAME]
    sentence_idx_column_name = form[SENTENCE_IDX_COLUMN_NAME_FIELD_NAME]
    word_column_name = form[WORD_COLUMN_NAME_FIELD_NAME]
    
    infile = request.files[DATASET_CSV_FIELD_NAME]
    
    key = "{}_{}".format(user_id, env_id)

    if key in TRAIN_QUEUES.keys():
        if TRAIN_QUEUES[key].is_alive():
            return compose_response("This function is disabled when training is in progress!", code=FAILURE_CODE)
    
    try:
        datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                                  [DATASET_ID_FIELD_NAME, DATASET_TYPE_FIELD_NAME, DATASET_PATH_FIELD_NAME], 
                                  [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                                  [user_id, env_id, dataset_name])
        
        if len(datasets) == 0:
            return compose_response("A dataset with this name doesn't exist!", code=FAILURE_CODE)
        
        existing_dataset = datasets[0]

        existing_dataset_id = existing_dataset[0]
        existing_dataset_type = existing_dataset[1]
        existing_dataset_path = existing_dataset[2]

        existing_dataset = pd.read_pickle(existing_dataset_path)
        imported_dataset = pd.read_csv(infile, keep_default_na=False)
        
        if text_column_name in imported_dataset.columns:
            imported_dataset.rename(columns={text_column_name: TEXT_FIELD_NAME})
        
        if sentence_idx_column_name in imported_dataset.columns:
            imported_dataset.rename(columns={sentence_idx_column_name: SENTENCE_IDX_FIELD_NAME})
        
        if word_column_name in imported_dataset.columns:
            imported_dataset.rename(columns={word_column_name: WORD_FIELD_NAME})
        
        needed_fields = []
        
        if existing_dataset_type == SLC_MODEL_TYPE:
            needed_fields += [TEXT_FIELD_NAME]
        elif existing_dataset_type == TLC_MODEL_TYPE:
            needed_fields += [SENTENCE_IDX_FIELD_NAME, WORD_FIELD_NAME]
        
        for needed_field in needed_fields:
            if needed_field not in imported_dataset.columns:
                return compose_response("Trying to import from an invalid dataframe!", code=FAILURE_CODE)

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
        
        if os.path.exists(existing_dataset_path):
            os.remove(existing_dataset_path)
        
        existing_dataset.to_pickle(existing_dataset_path)
    except:
        return compose_response("Trying to import from an invalid dataframe!", code=FAILURE_CODE)
    
    return compose_response("Examples imported succesfully!")

@app.route('/get_env_datasets', methods=['POST'])
def get_env_datasets():

    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)
    
    session = request.json[SESSION_FIELD_NAME]
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("No environment selected!", code=FAILURE_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    
    datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME,
                              [DATASET_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME, DATASET_TYPE_FIELD_NAME], 
                              [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME],
                              [user_id, env_id])
    
    data = [{"value" : {"id" : dataset[0], "name" : dataset[1], "type": dataset[2]}, "text" : dataset[1]} for dataset in datasets]

    return compose_response("Datasets fetched!", data)

@app.route('/get_dataset_examples', methods=['POST'])
def get_dataset_examples():
    
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    needed_fields = [DATASET_NAME_FIELD_NAME, "starting_index", "number_of_examples"]
    
    for needed_field in needed_fields:
        if needed_field not in json:
            return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    dataset_name = json[DATASET_NAME_FIELD_NAME]
    starting_index = int(json["starting_index"])
    number_of_examples_to_fetch = int(json["number_of_examples"])

    key = "{}_{}".format(user_id, env_id)

    if key in TRAIN_QUEUES.keys():
        if TRAIN_QUEUES[key].is_alive():
            return compose_response("This function is disabled when training is in progress!", code=FAILURE_CODE)
    
    try:
        datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                                  [DATASET_PATH_FIELD_NAME], 
                                  [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                                  [user_id, env_id, dataset_name])
        
        if len(datasets) == 0:
            return compose_response("A dataset with this name doesn't exist!", code=FAILURE_CODE)
        
        path_to_dataset = datasets[0][0]
        
        dataframe = pd.read_pickle(path_to_dataset)

        examples = list()
        fields = list()

        for column in dataframe.columns:

            column_name = str(column)

            if "Unnamed:" not in column_name:
                fields.append(column_name)
        
        counter = 0

        for (row_index, row) in dataframe.iterrows():

            if row_index < starting_index:
                continue

            currentElement = dict()

            for column in dataframe.columns:

                column_name = column
                
                if "Unnamed:" in str(column):
                    column_name = "index"
                
                currentElement[column_name] = row[column]
            
            if "index" not in currentElement:
                currentElement["index"] = row_index
            
            examples.append(currentElement)

            counter += 1

            if counter > number_of_examples_to_fetch:
                break
        
        data = {
            "examples": examples,
            "fields": fields,
        }

        return compose_response("Dataset examples fetched!", data)
    except Exception as ex:
        print(ex)
        return compose_response("Something went wrong, couldn't fetch the examples...", code=FAILURE_CODE)

@app.route('/modify_example_in_dataset', methods=['POST'])
def modify_example_in_dataset():
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    needed_fields = [DATASET_NAME_FIELD_NAME, "example_index", "example_text", "example_targets", "example_category"]
    
    for needed_field in needed_fields:
        if needed_field not in json:
            return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    dataset_name = json[DATASET_NAME_FIELD_NAME]
    example_index = json["example_index"]
    example_text = json["example_text"]
    example_targets = json["example_targets"]
    example_category = json["example_category"]

    key = "{}_{}".format(user_id, env_id)

    if key in TRAIN_QUEUES.keys():
        if TRAIN_QUEUES[key].is_alive():
            return compose_response("This function is disabled when training is in progress!", code=FAILURE_CODE)
    
    try:
        datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                                  [DATASET_PATH_FIELD_NAME], 
                                  [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                                  [user_id, env_id, dataset_name])
        
        if len(datasets) == 0:
            return compose_response("A dataset with this name doesn't exist!", code=FAILURE_CODE)
        
        path_to_dataset = datasets[0][0]
        
        dataframe = pd.read_pickle(path_to_dataset)

        new_row = dict()

        new_row[TEXT_FIELD_NAME] = example_text
        new_row[EXAMPLE_CATEGORY_FIELD_NAME] = example_category
        
        for new_example_target_data in example_targets:

            target_name = new_example_target_data["targetName"]
            target_value = new_example_target_data["targetValue"]

            if target_name not in dataframe.columns:
                dataframe[target_name] = EMPTY_TARGET_VALUE
            
            new_row[target_name] = target_value
        
        columns_to_drop = []

        for column in dataframe.columns:
            
            if str(column) not in new_row.keys():
                new_row[str(column)] = EMPTY_TARGET_VALUE

            if "Unnamed:" in str(column):
                columns_to_drop.append(str(column))
        
        dataframe.loc[example_index] = new_row

        dataframe.drop(columns=columns_to_drop, inplace=True)
        
        if os.path.exists(path_to_dataset):
            os.remove(path_to_dataset)

        dataframe.to_pickle(path_to_dataset)

        return compose_response("Example modified!")
    except Exception as ex:
        print(ex)
        return compose_response("Something went wrong, couldn't add the example the dataset...", code=FAILURE_CODE)

@app.route('/append_example_to_dataset', methods=['POST'])
def append_example_to_dataset():
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    needed_fields = [DATASET_NAME_FIELD_NAME, "example_text", "example_targets", "example_category"]
    
    for needed_field in needed_fields:
        if needed_field not in json:
            return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    dataset_name = json[DATASET_NAME_FIELD_NAME]
    example_text = json["example_text"]
    example_targets = json["example_targets"]
    example_category = json["example_category"]

    key = "{}_{}".format(user_id, env_id)

    if key in TRAIN_QUEUES.keys():
        if TRAIN_QUEUES[key].is_alive():
            return compose_response("This function is disabled when training is in progress!", code=FAILURE_CODE)
    
    try:
        datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                                  [DATASET_PATH_FIELD_NAME], 
                                  [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                                  [user_id, env_id, dataset_name])
        
        if len(datasets) == 0:
            return compose_response("A dataset with this name doesn't exist!", code=FAILURE_CODE)
        
        path_to_dataset = datasets[0][0]
        
        dataframe = pd.read_pickle(path_to_dataset)

        new_row = dict()

        new_row[TEXT_FIELD_NAME] = example_text
        new_row[EXAMPLE_CATEGORY_FIELD_NAME] = example_category
        
        for new_example_target_data in example_targets:

            target_name = new_example_target_data["targetName"]
            target_value = new_example_target_data["targetValue"]

            if target_name not in dataframe.columns:
                dataframe[target_name] = EMPTY_TARGET_VALUE
            
            new_row[target_name] = target_value
        
        columns_to_drop = []

        for column in dataframe.columns:
            
            if str(column) not in new_row.keys():
                new_row[str(column)] = EMPTY_TARGET_VALUE

            if "Unnamed:" in str(column):
                columns_to_drop.append(str(column))
        
        dataframe.loc[len(dataframe) + 1] = new_row

        dataframe.drop(columns=columns_to_drop, inplace=True)
        
        if os.path.exists(path_to_dataset):
            os.remove(path_to_dataset)

        dataframe.to_pickle(path_to_dataset)

        return compose_response("Example added!")
    except Exception as ex:
        print(ex)
        return compose_response("Something went wrong, couldn't add the example the dataset...", code=FAILURE_CODE)

@app.route('/remove_example_from_dataset', methods=['POST'])
def remove_example_from_dataset():
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    needed_fields = [DATASET_NAME_FIELD_NAME, "example_index"]
    
    for needed_field in needed_fields:
        if needed_field not in json:
            return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    dataset_name = json[DATASET_NAME_FIELD_NAME]
    example_index = json["example_index"]

    key = "{}_{}".format(user_id, env_id)

    if key in TRAIN_QUEUES.keys():
        if TRAIN_QUEUES[key].is_alive():
            return compose_response("This function is disabled when training is in progress!", code=FAILURE_CODE)
    
    try:
        datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                                  [DATASET_PATH_FIELD_NAME], 
                                  [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                                  [user_id, env_id, dataset_name])
        
        if len(datasets) == 0:
            return compose_response("A dataset with this name doesn't exist!", code=FAILURE_CODE)
        
        path_to_dataset = datasets[0][0]
        
        dataframe = pd.read_pickle(path_to_dataset)
        
        columns_to_drop = []

        for column in dataframe.columns:
            
            if "Unnamed:" in str(column):
                columns_to_drop.append(str(column))
        
        print(example_index)

        dataframe.drop(example_index, inplace=True)
        
        print(dataframe.head())

        dataframe.drop(columns=columns_to_drop, inplace=True)
        
        if os.path.exists(path_to_dataset):
            os.remove(path_to_dataset)

        dataframe.to_pickle(path_to_dataset)

        return compose_response("Example removed!")
    except Exception as ex:
        print(ex)
        return compose_response("Something went wrong, couldn't remove the example the dataset...", code=FAILURE_CODE)

#endregion

#region TRAINING QUEUE HANDLING

class TrainQueueProgressInfo:
    
    def __init__(self):
        self.current_session_index = 0
        self.current_session_model_name = ""
        self.current_session_dataset_name = ""
        self.current_session_epoch = 0
        self.current_session_epochs_left = 0
        self.wants_to_stop = False
        self.stopped_prematurely = False
    
    def get_payload(self):

        payload = {
            "current_session_index" : self.current_session_index,
            "current_session_model_name" : self.current_session_model_name,
            "current_session_dataset_name" : self.current_session_dataset_name,
            "current_session_epoch" : self.current_session_epoch,
            "current_session_epochs_left" : self.current_session_epochs_left
        }

        return payload

@app.route('/add_to_train_queue', methods=['POST'])
def add_model_to_train_queue():
    
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    needed_form_fields = [MODEL_NAME_FIELD_NAME, DATASET_NAME_FIELD_NAME, 
                          TARGETS_FIELD_NAME, NUM_OF_EPOCHS_FIELD_NAME, 
                          BATCH_SIZE_FIELD_NAME]
    
    for needed_form_field in needed_form_fields:
        if needed_form_field not in json:
            return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    username = session[USERNAME_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    env_name = session[ENV_NAME_FIELD_NAME]
    model_name = json[MODEL_NAME_FIELD_NAME]
    dataset_name = json[DATASET_NAME_FIELD_NAME]
    target = json[TARGETS_FIELD_NAME]
    num_of_epochs = json[NUM_OF_EPOCHS_FIELD_NAME]
    batch_size = json[BATCH_SIZE_FIELD_NAME]
    
    key = "{}_{}".format(user_id, env_id)

    if key in TRAIN_QUEUES.keys():
        if TRAIN_QUEUES[key].is_alive():
            return compose_response("This function is disabled when training is in progress!", code=FAILURE_CODE)
    
    try:
        models = select_from_db(ALIVE_DB_MODELS_TABLE_NAME, 
                                [MODEL_ID_FIELD_NAME, MODEL_TYPE_FIELD_NAME, FINETUNABLE_FIELD_NAME], 
                                [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME], 
                                [user_id, env_id, model_name])
        
        if len(models) == 0:
            return compose_response("A model with this name doesn't exist!", code=FAILURE_CODE)
        
        model_id = models[0][0]
        model_type = models[0][1]
        model_finetunable = models[0][2]
        
        datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                                  [DATASET_ID_FIELD_NAME, DATASET_TYPE_FIELD_NAME], 
                                  [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                                  [user_id, env_id, dataset_name])
        
        if len(datasets) == 0:
            return compose_response("A dataset with this name doesn't exist!", code=FAILURE_CODE)
        
        dataset_id = datasets[0][0]
        dataset_type = datasets[0][1]

        if model_type != dataset_type:
            return compose_response("Can't add to train queue, dataset type is invalid!", code=FAILURE_CODE)
        
        queue_in_this_env = select_from_db(ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME, 
                                           [QUEUE_INDEX_FIELD_NAME, MODEL_ID_FIELD_NAME], 
                                           [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME], 
                                           [user_id, env_id])
        
        max_id = 0

        for training_session in queue_in_this_env:
            
            if training_session[0] > max_id:
                max_id = training_session[0]
            
            if model_id == training_session[1] and not model_finetunable:
                return compose_response("Can't add this model, it is already in queue and not finetunable!", code=FAILURE_CODE)
            
        new_id = max_id + 1

        path_to_env = USERS_DATA_FOLDER + username + "/" + ENVIRONMENTS_FOLDER_NAME + "/" + env_name + "/"
        path_to_training_sessions = path_to_env + TRAINING_SESSIONS_FOLDER_NAME + "/"
        checkpoint_name = str(model_id) + "_" + str(dataset_id) + "_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        checkpoint_path = path_to_training_sessions + checkpoint_name + "/"
        
        insert_into_db(ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, QUEUE_INDEX_FIELD_NAME, 
                        MODEL_ID_FIELD_NAME, DATASET_ID_FIELD_NAME, TARGETS_FIELD_NAME, 
                        NUM_OF_EPOCHS_FIELD_NAME, BATCH_SIZE_FIELD_NAME, CHECKPOINT_PATH_FIELD_NAME], 
                        [user_id, env_id, new_id, 
                         model_id, dataset_id, target, 
                         num_of_epochs, batch_size, checkpoint_path])
    except:
        return compose_response("Something went wrong, couldn't add to train queue...", code=FAILURE_CODE)
    
    return compose_response("Successfully added to train queue!")

@app.route('/remove_from_train_queue', methods=['POST'])
def remove_session_from_train_queue():
    
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    json = request.json
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    if QUEUE_INDEX_FIELD_NAME not in json:
        return compose_response("Didn't recieve needed fields!", code=MISSING_FIELDS_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    queue_index = json[QUEUE_INDEX_FIELD_NAME]
    
    key = "{}_{}".format(user_id, env_id)

    if key in TRAIN_QUEUES.keys():
        if TRAIN_QUEUES[key].is_alive():
            return compose_response("This function is disabled when training is in progress!", code=FAILURE_CODE)
    
    try:
        training_sessions = select_from_db(ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME, 
                                           [CHECKPOINT_PATH_FIELD_NAME], 
                                           [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, QUEUE_INDEX_FIELD_NAME], 
                                           [user_id, env_id, queue_index])
        
        if len(training_sessions) == 0:
            return compose_response("There is no training session at this index!", code=FAILURE_CODE)
        
        checkpoint_path = training_sessions[0][0]

        delete_from_db(ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME, 
                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, QUEUE_INDEX_FIELD_NAME], 
                       [user_id, env_id, queue_index])
        
        if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
            shutil.rmtree(checkpoint_path)
    except:
        return compose_response("Something went wrong, couldn't remove the training session from the queue...", code=FAILURE_CODE)
    
    return compose_response("Succesfully removed training session!")

@app.route('/start_train', methods=['POST'])
def start_train():
    
    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)

    session = request.json[SESSION_FIELD_NAME]
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("Environment not selected!", code=FAILURE_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    
    key = "{}_{}".format(user_id, env_id)

    if key in TRAIN_QUEUES.keys():
        if TRAIN_QUEUES[key].is_alive():
            return compose_response("Training is already in progress!", code=FAILURE_CODE)
        else:
            del(TRAIN_QUEUES[key])
    
    TRAIN_QUEUES_PROGRESS_INFOS[key] = TrainQueueProgressInfo()
    
    lambda_training_function = lambda : train_queue(user_id, env_id, TRAIN_QUEUES_PROGRESS_INFOS[key])
    training_thread = Thread(target=lambda_training_function, daemon=True, name='Monitor')
    training_thread.start()

    TRAIN_QUEUES[key] = training_thread

    return compose_response("Training started succesfully!")

def train_queue(user_id:int, env_id:int, progress_info:TrainQueueProgressInfo):
    
    connection = mysqlconn.connect(user=ALIVE_DB_ADMIN_USERNAME, password=ALIVE_DB_ADMIN_PASSWORD, database=ALIVE_DB_NAME)
    
    queue_in_this_env = select_from_db(ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME, 
                                       [QUEUE_INDEX_FIELD_NAME, 
                                        MODEL_ID_FIELD_NAME, DATASET_ID_FIELD_NAME, 
                                        TARGETS_FIELD_NAME, NUM_OF_EPOCHS_FIELD_NAME,
                                        BATCH_SIZE_FIELD_NAME, CHECKPOINT_PATH_FIELD_NAME], 
                                       [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME], 
                                       [user_id, env_id], 
                                       connection)
    
    if len(queue_in_this_env) == 0:
        print("Nothing on train queue!")
        return
    
    queue_in_this_env.sort(key=(lambda session:session[0]))

    training_sessions = Queue()

    for training_session in queue_in_this_env:
        training_sessions.put(training_session)
    
    while not training_sessions.empty():

        print("\nQUEUE SIZE: {}\n".format(training_sessions.qsize()))

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
                                    [MODEL_PATH_FIELD_NAME, MODEL_TYPE_FIELD_NAME, MODEL_NAME_FIELD_NAME], 
                                    [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, MODEL_ID_FIELD_NAME], 
                                    [user_id, env_id, model_id], 
                                    connection)
            
            if len(models) == 0:
                print("Couldn't find the model specified in this train session!")
                continue
            
            model = models[0]

            path_to_model = model[0]
            model_type = model[1]
            model_name = model[2]
            
            datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME, 
                                      [DATASET_PATH_FIELD_NAME, DATASET_NAME_FIELD_NAME], 
                                      [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME, DATASET_ID_FIELD_NAME], 
                                      [user_id, env_id, dataset_id], 
                                      connection)
            
            if len(datasets) == 0:
                print("Couldn't find the dataset specified in this train session!")
                continue
            
            dataset = datasets[0]

            path_to_dataset = dataset[0]
            dataset_name = dataset[1]

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
            
            testfeatures, testtargets = data.get_test()

            progress_info.current_session_index = current_queue_index
            progress_info.current_session_model_name = model_name
            progress_info.current_session_dataset_name = dataset_name
            progress_info.current_session_epochs_left = epochs_left

            epochs_updating_callback = UpdateDBCallback(user_id, env_id, 
                                                        current_queue_index, progress_info, 
                                                        connection)
            additional_callbacks = [epochs_updating_callback]
            
            history, ending_time = loaded_model.train(data, epochs_left, batch_size, checkpoint_path, additional_callbacks)

            if not progress_info.stopped_prematurely:
                loaded_model.save(path_to_model, True)
                shutil.rmtree(checkpoint_path)

                graph_path = path_to_model + TRAINING_GRAPHS_FOLDER_NAME + "/"

                try:
                    loaded_model.save_train_history_graph(history.history, ending_time, graph_path)
                except:
                    print("Couldn't save graph...")
                
                try:
                    print(loaded_model.evaluate(testfeatures, testtargets))
                except:
                    print("Couldn't evaluate...")
                
                delete_from_db(ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME, 
                            [ENV_ID_FIELD_NAME, QUEUE_INDEX_FIELD_NAME], 
                            [env_id, current_queue_index], 
                            connection)
            else:
                print("Training stopped prematurely...")
                break
        except Exception as ex:
            print(ex)
            continue
    
    key = "{}_{}".format(user_id, env_id)
    
    del(TRAIN_QUEUES[key])
    del(TRAIN_QUEUES_PROGRESS_INFOS[key])
    print("\nTraining is over.\n")

@app.route('/stop_train', methods=['POST'])
def stop_train():
    
    try:
        if SESSION_FIELD_NAME not in request.json:
            return compose_response("Couldn't find session!", code=FAILURE_CODE)

        session = request.json[SESSION_FIELD_NAME]
        
        if USER_ID_FIELD_NAME not in session:
            return compose_response("User not logged!", code=FAILURE_CODE)
        
        if ENV_ID_FIELD_NAME not in session:
            return compose_response("Environment not selected!", code=FAILURE_CODE)
            
        user_id = session[USER_ID_FIELD_NAME]
        env_id = session[ENV_ID_FIELD_NAME]
        
        key = "{}_{}".format(user_id, env_id)

        if key not in TRAIN_QUEUES.keys():
            return compose_response("Training is not running!", code=FAILURE_CODE)
        
        if key in TRAIN_QUEUES_PROGRESS_INFOS:
            TRAIN_QUEUES_PROGRESS_INFOS[key].wants_to_stop = True
        
        return compose_response("Stopped training progress!")
    except Exception as ex:
        print(ex)
        return compose_response("Couldn't stop the training...")

@app.route('/get_train_queue', methods=['POST'])
def get_train_queue():

    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)
    
    session = request.json[SESSION_FIELD_NAME]
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("No environment selected!", code=FAILURE_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    
    try:
        training_sessions = select_from_db(ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME,
                                           [QUEUE_INDEX_FIELD_NAME, MODEL_ID_FIELD_NAME, DATASET_ID_FIELD_NAME, 
                                            NUM_OF_EPOCHS_FIELD_NAME, BATCH_SIZE_FIELD_NAME], 
                                           [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME],
                                           [user_id, env_id])
        
        models = select_from_db(ALIVE_DB_MODELS_TABLE_NAME,
                                [MODEL_ID_FIELD_NAME, MODEL_NAME_FIELD_NAME],
                                [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME],
                                [user_id, env_id])
        
        datasets = select_from_db(ALIVE_DB_DATASETS_TABLE_NAME,
                                  [DATASET_ID_FIELD_NAME, DATASET_NAME_FIELD_NAME],
                                  [USER_ID_FIELD_NAME, ENV_ID_FIELD_NAME],
                                  [user_id, env_id])
        
        model_names = dict()
        dataset_names = dict()

        for model in models:
            model_id = model[0]
            model_name = model[1]
            model_names[model_id] = model_name
        
        for dataset in datasets:
            dataset_id = dataset[0]
            dataset_name = dataset[1]
            dataset_names[dataset_id] = dataset_name
        
        data = list()

        for training_session in training_sessions:
            id = training_session[0]
            model_id = training_session[1]
            dataset_id = training_session[2]
            model_name = model_names[model_id]
            dataset_name = dataset_names[dataset_id]
            num_of_epochs = training_session[3]
            batch_size = training_session[4]

            value = {"id" : id, "model_name" : model_name, "dataset_name" : dataset_name, 
                     "num_of_epochs" : num_of_epochs, "batch_size" : batch_size}
            
            text = "{} - {}".format(model_name, dataset_name)

            data.append({"value" : value, "text" : text})
            
        return compose_response("Train queue fetched!", data)
    except:
        return compose_response("Couldn't fetch train queue...", code=FAILURE_CODE)

@app.route('/get_train_queue_progress_info', methods=['POST'])
def get_train_queue_progress_info():

    if SESSION_FIELD_NAME not in request.json:
        return compose_response("Couldn't find session!", code=FAILURE_CODE)
    
    session = request.json[SESSION_FIELD_NAME]
    
    if USER_ID_FIELD_NAME not in session:
        return compose_response("User not logged!", code=FAILURE_CODE)
    
    if ENV_ID_FIELD_NAME not in session:
        return compose_response("No environment selected!", code=FAILURE_CODE)
    
    user_id = session[USER_ID_FIELD_NAME]
    env_id = session[ENV_ID_FIELD_NAME]
    
    key = "{}_{}".format(user_id, env_id)
    
    if key in TRAIN_QUEUES:
        if isinstance(TRAIN_QUEUES[key], Thread):
            if TRAIN_QUEUES[key].is_alive():
                if key in TRAIN_QUEUES_PROGRESS_INFOS:
                    return compose_response("Training progress fetched!", TRAIN_QUEUES_PROGRESS_INFOS[key].get_payload())
            else:
                return compose_response("Training terminated!")
    
    return compose_response("Couldn't fetch training progress...", code=FAILURE_CODE)

#endregion

#region OTHER APIS

@app.route('/fetch_available_model_types', methods=['POST'])
def fetch_available_model_types():
    
    data = [{"value" : model_type, "text" : model_type} for model_type in MODEL_TYPES]

    return compose_response("Model types fetched!", data)

@app.route('/fetch_available_base_models', methods=['POST'])
def fetch_available_base_models():

    base_models = get_available_base_models()
    
    data = [{"value" : base_model, "text" : base_model} for base_model in base_models]

    return compose_response("Base models fetched!", data)

@app.route('/fetch_available_example_categories', methods=['POST'])
def fetch_available_example_categories():

    data = [{"value" : example_category, "text" : example_category} for example_category in EXAMPLE_CATEGORIES]

    return compose_response("Example categories fetched!", data)

#endregion

def initialize_server():

    path_to_users_data = USERS_DATA_FOLDER

    if not os.path.exists(path_to_users_data):
        os.makedirs(path_to_users_data)

#region DB UTILITIES

def select_from_db(table_name:str, needed_fields:list=None, 
                   given_fields:list=None, given_values:list=None, 
                   connection:mysqlconn.MySQLConnection=None):
    
    if connection == None:
        connection = DB_CONNECTION

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
    
    cursor = connection.cursor(buffered=True)
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    
    return results

def insert_into_db(table_name:str, given_fields:list, given_values:list, 
                   connection:mysqlconn.MySQLConnection=None):
    
    if connection == None:
        connection = DB_CONNECTION
    
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

    cursor = connection.cursor(buffered=True)
    cursor.execute(query)
    connection.commit()
    cursor.close()

def update_db(table_name:str, 
              fields_to_update:list=None, updated_values:list=None,
              given_fields:list=None, given_values:list=None, 
              connection:mysqlconn.MySQLConnection=None):
    
    if connection == None:
        connection = DB_CONNECTION
    
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
    
    cursor = connection.cursor(buffered=True)
    cursor.execute(query)
    connection.commit()
    cursor.close()

def delete_from_db(table_name:str, given_fields:list=None, given_values:list=None, 
                   connection:mysqlconn.MySQLConnection=None):
    
    if connection == None:
        connection = DB_CONNECTION
    
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
    
    cursor = connection.cursor(buffered=True)
    cursor.execute(query)
    connection.commit()
    cursor.close()

def execute_custom_update_query(query, connection:mysqlconn.MySQLConnection=None):
    
    if connection == None:
        connection = DB_CONNECTION
    
    cursor = connection.cursor(buffered=True)
    cursor.execute(query)
    connection.commit()
    cursor.close()

#endregion

class UpdateDBCallback(tf.keras.callbacks.Callback):

    def __init__(self, user_id:int, env_id:int, current_queue_index:int, progress_info:TrainQueueProgressInfo,
                 connection:mysqlconn.MySQLConnection=None):
        super().__init__()
        self.__user_id = user_id
        self.__env_id = env_id
        self.__current_queue_index = current_queue_index
        self.__progress_info = progress_info

        if connection == None:
            connection = DB_CONNECTION
        
        self.__connection = connection
    
    def on_batch_begin(self, batch, logs=None):

        if self.__progress_info.wants_to_stop:
            self.__progress_info.stopped_prematurely = True
            self.model.stop_training = True
    
    def on_epoch_begin(self, epoch, logs=None):
        self.__progress_info.current_session_epoch += 1
    
    def on_epoch_end(self, epoch, logs=None):

        user_id = self.__user_id
        env_id = self.__env_id
        current_queue_index = self.__current_queue_index

        update_epochs_left_query = "UPDATE " + ALIVE_DB_TRAINING_SESSIONS_TABLE_NAME + " SET "
        update_epochs_left_query += NUM_OF_EPOCHS_FIELD_NAME + " = " + NUM_OF_EPOCHS_FIELD_NAME + " - 1 "
        update_epochs_left_query += " WHERE " + USER_ID_FIELD_NAME + " = " + str(user_id) + " AND "
        update_epochs_left_query += ENV_ID_FIELD_NAME + " = " + str(env_id) + " AND "
        update_epochs_left_query += QUEUE_INDEX_FIELD_NAME + " = " + str(current_queue_index)
        
        cursor = self.__connection.cursor(buffered=True)
        cursor.execute(update_epochs_left_query)
        self.__connection.commit()
        cursor.close()

#region FRONT COMMUNICATION

SUCCESS_CODE = 1
FAILURE_CODE = 1000
MISSING_FIELDS_CODE = 1001
RETURNING_ERRORS_CODE = 1002

def compose_response(message:str, data=None, code:int=SUCCESS_CODE):
    return jsonify({
        "code" : code,
        "message" : message,
        "data" : data
    })

#endregion

if __name__ == "__main__":
    initialize_server()
    #app.secret_key = os.urandom(12)
    app.run(debug=False,host='0.0.0.0', port=5000)
