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
    if "e" in sys.argv or "m" in sys.argv or "d" in sys.argv:
        cursor.execute("DELETE FROM users_environments")
    if "m" in sys.argv:
        cursor.execute("DELETE FROM environments_models")
    if "d" in sys.argv:
        cursor.execute("DELETE FROM environments_datasets")

db_connection.commit()
cursor.close()
