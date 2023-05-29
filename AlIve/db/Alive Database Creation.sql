CREATE DATABASE IF NOT EXISTS alive_db;

USE alive_db;

CREATE TYPE IF NOT EXISTS MODEL_TYPE AS ENUM("SLCM", "TLCM")

CREATE TABLE IF NOT EXISTS alive_users ( 
    user_id INT(11) AUTO_INCREMENT NOT NULL PRIMARY KEY, 
    username VARCHAR(100) NOT NULL UNIQUE, 
    user_password VARCHAR(500) NOT NULL, 
    email VARCHAR(200)
);

ALTER TABLE alive_users
ADD CONSTRAINT IF NOT EXISTS email_validation
CHECK ((email IS NULL) 
       OR 
       (email REGEXP "^[a-zA-Z0-9][a-zA-Z0-9.!#$%&'*+-/=?^_`{|}~]*?[a-zA-Z0-9._-]?@[a-zA-Z0-9][a-zA-Z0-9._-]*?[a-zA-Z0-9]?\\.[a-zA-Z]{2,63}$") 
      );

CREATE TABLE IF NOT EXISTS users_environments ( 
    user_id INT(11) NOT NULL,
    env_id INT(11) NOT NULL, 
    env_name VARCHAR(100) NOT NULL,
    PRIMARY KEY (user_id, env_id)
    UNIQUE KEY unique_env_name (user_id, env_name)
);

ALTER TABLE users_environments
ADD CONSTRAINT IF NOT EXISTS environments_to_users
FOREIGN KEY (user_id) REFERENCES alive_users(user_id);

CREATE TABLE IF NOT EXISTS environments_models ( 
    user_id INT(11) NOT NULL,
    env_id INT(11) NOT NULL, 
    model_id INT(11) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_path VARCHAR(1000) NOT NULL,
    model_type MODEL_TYPE NOT NULL,
    PRIMARY KEY (user_id, env_id, model_id)
    UNIQUE KEY unique_model_name (user_id, env_id, model_name)
);

ALTER TABLE environments_models
ADD CONSTRAINT IF NOT EXISTS models_to_environments
FOREIGN KEY (user_id, env_id) REFERENCES users_environments(user_id, env_id);

CREATE TABLE IF NOT EXISTS environments_datasets ( 
    user_id INT(11) NOT NULL,
    env_id INT(11) NOT NULL, 
    dataset_id INT(11) NOT NULL,
    dataset_name VARCHAR(100) NOT NULL,
    dataset_path VARCHAR(1000) NOT NULL,
    dataset_type MODEL_TYPE NOT NULL,
    PRIMARY KEY (user_id, env_id, dataset_id)
    UNIQUE KEY unique_dataset_name (user_id, env_id, dataset_name)
);

ALTER TABLE environments_datasets
ADD CONSTRAINT IF NOT EXISTS datasets_to_environments
FOREIGN KEY (user_id, env_id) REFERENCES users_environments(user_id, env_id);
