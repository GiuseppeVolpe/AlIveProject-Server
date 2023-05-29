import sys
import mysql.connector as mysqlconn

db_connection = mysqlconn.connect(user='GiuseppeVolpe', password='password', database='alive_db')

cursor = db_connection.cursor()

if len(sys.argv) == 0:
    cursor.execute("DELETE FROM alive_users")
    cursor.execute("DELETE FROM users_environments")
    cursor.execute("DELETE FROM environments_models")
    cursor.execute("DELETE FROM environments_datasets")
else:
    if "e" in sys.argv or "m" in sys.argv or "d" in sys.argv:
        cursor.execute("DELETE FROM users_environments")
    if "m" in sys.argv:
        cursor.execute("DELETE FROM environments_models")
    if "d" in sys.argv:
        cursor.execute("DELETE FROM environments_datasets")
