import mysql.connector as mysqlconn

db_connection = mysqlconn.connect(user='GiuseppeVolpe', password='password', database='alive_db')

cursor = db_connection.cursor()
cursor.execute("DELETE FROM alive_users")
cursor.execute("DELETE FROM users_environments")
cursor.execute("DELETE FROM environments_models")
cursor.execute("DELETE FROM environments_datasets")
