CREATE DATABASE IF NOT EXISTS alive_db;

USE alive_db;

CREATE TABLE IF NOT EXISTS alive_users ( 
    user_id INT(11) NOT NULL PRIMARY KEY AUTO_INCREMENT, 
    username VARCHAR(100) NOT NULL UNIQUE, 
    user_password VARCHAR(500) NOT NULL, 
    user_email VARCHAR(200)
);

ALTER TABLE alive_users
ADD CONSTRAINT user_email_validation
CHECK ((user_email IS NULL) 
       OR 
       (user_email REGEXP "^[a-zA-Z0-9][a-zA-Z0-9.!#$%&'*+-/=?^_`{|}~]*?[a-zA-Z0-9._-]?@[a-zA-Z0-9][a-zA-Z0-9._-]*?[a-zA-Z0-9]?\\.[a-zA-Z]{2,63}$") 
      );

CREATE TABLE IF NOT EXISTS users_environments ( 
    user_id INT(11) NOT NULL,
    env_id INT(11) NOT NULL, 
    env_name VARCHAR(100) NOT NULL,
    public BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (user_id, env_id),
    UNIQUE (user_id, env_name)
);

ALTER TABLE users_environments
ADD CONSTRAINT environments_to_users
FOREIGN KEY (user_id) REFERENCES alive_users(user_id);

CREATE TABLE IF NOT EXISTS environments_models ( 
    user_id INT(11) NOT NULL,
    env_id INT(11) NOT NULL, 
    model_id INT(11) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_path VARCHAR(1000) NOT NULL,
    model_type ENUM("SLCM", "TLCM") NOT NULL,
    finetunable BOOLEAN NOT NULL DEFAULT FALSE,
    public BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (user_id, env_id, model_id),
    UNIQUE (user_id, env_id, model_name)
);

ALTER TABLE environments_models
ADD CONSTRAINT models_to_environments
FOREIGN KEY (user_id, env_id) REFERENCES users_environments(user_id, env_id);

CREATE TABLE IF NOT EXISTS environments_datasets ( 
    user_id INT(11) NOT NULL,
    env_id INT(11) NOT NULL, 
    dataset_id INT(11) NOT NULL,
    dataset_name VARCHAR(100) NOT NULL,
    dataset_path VARCHAR(1000) NOT NULL,
    dataset_type ENUM("SLCM", "TLCM") NOT NULL,
    public BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (user_id, env_id, dataset_id),
    UNIQUE (user_id, env_id, dataset_name)
);

ALTER TABLE environments_datasets
ADD CONSTRAINT datasets_to_environments
FOREIGN KEY (user_id, env_id) REFERENCES users_environments(user_id, env_id);

CREATE TABLE IF NOT EXISTS training_sessions (
    user_id INT(11) NOT NULL,
    env_id INT(11) NOT NULL,
    queue_index INT(11) NOT NULL,
    model_id INT(11) NOT NULL,
    dataset_id INT(11) NOT NULL,
    targets VARCHAR(1000) NOT NULL,
    epochs_left INT(11) NOT NULL,
    batch_size INT(11) NOT NULL,
    checkpoint_path VARCHAR(1000) NOT NULL,
    PRIMARY KEY (user_id, env_id, queue_index),
    UNIQUE (user_id, env_id, model_id)
);

ALTER TABLE training_sessions
ADD CONSTRAINT sessions_to_models
FOREIGN KEY (user_id, env_id, model_id) REFERENCES users_environments(user_id, env_id, model_id);

ALTER TABLE training_sessions
ADD CONSTRAINT sessions_to_datasets
FOREIGN KEY (user_id, env_id, dataset_id) REFERENCES users_environments(user_id, env_id, dataset_id);
