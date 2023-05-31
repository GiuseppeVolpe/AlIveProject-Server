import sys
import mysql.connector as mysqlconn

db_connection = mysqlconn.connect(user='GiuseppeVolpe', password='password', database='alive_db')

cursor = db_connection.cursor()

if len(sys.argv) <= 1:
    cursor.execute("DELETE FROM environments_models")
    cursor.execute("DELETE FROM environments_datasets")
    cursor.execute("DELETE FROM users_environments")
    cursor.execute("DELETE FROM alive_users")
else:
    if "m" in sys.argv:
        cursor.execute("DELETE FROM environments_models")
    if "d" in sys.argv:
        cursor.execute("DELETE FROM environments_datasets")
    if "e" in sys.argv and "m" in sys.argv and "d" in sys.argv:
        cursor.execute("DELETE FROM users_environments")

db_connection.commit()
cursor.close()
