import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="ecopackAI",
        user="postgres",
        password="shash",
        port="5432"
    )

