CREATE DATABASE IF NOT EXISTS alive_db;

USE alive_db;

CREATE TABLE IF NOT EXISTS alive_users ( 
uid INT(11) AUTO_INCREMENT PRIMARY KEY, 
username VARCHAR(100), 
password VARCHAR(200), 
email VARCHAR(200)
);