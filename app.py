from flask import Flask, render_template, request
import pandas as pd
import sqlite3
import os
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

app = Flask(__name__)

# ---------- DATABASE ----------
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        usn TEXT,
        branch TEXT,
        score REAL
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------- LOAD DATA ----------
data = pd.read_csv("student_data.csv")

X = data.drop("final_score", axis=1)
y = data["final_score"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        name = request.form["name"]
        usn = request.form["usn"]
        branch = request.form["branch"]

        prev_cgpa = float(request.form["prev_cgpa"])
        prev_sgpa = float(request.form["prev_sgpa"])
        curr_cgpa = float(request.form["curr_cgpa"])
        curr_sgpa = float(request.form["curr_sgpa"])
        attendance = float(request.form["attendance"])
        project = float(request.form["project"])

        features = [[prev_cgpa, prev_sgpa, curr_cgpa, curr_sgpa, attendance, project]]
        prediction = model.predict(features)[0]

        # Save to DB
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute(
            "INSERT INTO students (name, usn, branch, score) VALUES (?, ?, ?, ?)",
            (name, usn, branch, float(prediction))
        )
        conn.commit()
        conn.close()

        # Create graph
        plt.figure()
        plt.bar(["Predicted Score"], [prediction])
        plt.savefig("static/graph.png")
        plt.close()

        return render_template("index.html",
                            prediction_text=f"Predicted Score: {round(prediction,2)}",
                            graph="graph.png")

    except Exception as e:
        return str(e)

@app.route("/students")
def students():
    conn = sqlite3.connect("database.db")
    df = pd.read_sql_query("SELECT * FROM students", conn)
    conn.close()

    return render_template("students.html", tables=df.to_dict(orient="records"))

# ---------- RUN ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)