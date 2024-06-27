from flask import Flask, request, jsonify, g
from flask_cors import CORS
import psycopg2
import os
import time
from psycopg2 import OperationalError

app = Flask(__name__)
CORS(app, origins=["http://localhost:8501"])

# PostgreSQL database setup
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://p4:p4@db:5432/p4")

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        with app.open_resource('schema.sql', mode='r') as f:
            sql_commands = f.read().split(';')
            for command in sql_commands:
                if command.strip():
                    cursor.execute(command)
        db.commit()

def get_db():
    retries = 5
    while retries > 0:
        try:
            connection = psycopg2.connect(DATABASE_URL)
            return connection
        except OperationalError as e:
            print(f"Database connection failed: {e}")
            retries -= 1
            print(f"Retrying in 5 seconds... ({retries} retries left)")
            time.sleep(5)
    raise Exception("Could not connect to the database after several retries")

@app.teardown_appcontext
def close_connection(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

@app.route("/save", methods=["POST"])
def upload_pdf():
    try:
        pdf_id = request.form.get("pdf_id")
        data = request.form.get("data")

        if not pdf_id or not data:
            return jsonify({"error": "Missing pdf_id or data"}), 400

        save_pdf_to_database(pdf_id, data)
        return jsonify({"message": "PDF data saved successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def save_pdf_to_database(pdf_id, data):
    db = get_db()
    cursor = db.cursor()
    cursor.execute('INSERT INTO pdf_files (pdf_name, pdf_information) VALUES (%s, %s)', (pdf_id, data))
    db.commit()

if __name__ == "__main__":
    init_db()
    app.run(port=5000, debug=True)
