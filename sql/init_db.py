import sqlite3

conn = sqlite3.connect('db/webserver.db')

with open('sql/schema.sql') as file:
    conn.executescript(file.read())

conn.commit()
conn.close()