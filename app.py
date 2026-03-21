from flask import Flask, render_template, request
import pandas as pd
import sqlite3
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io, base64

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
        score REAL,
        performance TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ---------- ML MODEL ----------
data = pd.read_csv("StudentsPerformance.csv")

X = data.drop("math score", axis=1)
y = data["math score"]

cat_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

model = Pipeline([
    ("pre", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42))
])

model.fit(X, y)

# ---------- ROUTES ----------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        name = request.form["name"]
        usn = request.form["usn"]
        branch = request.form["branch"]

        prev_cgpa = float(request.form["prev_cgpa"])
        prev_sgpa = float(request.form["prev_sgpa"])
        curr_cgpa = float(request.form["curr_cgpa"])
        curr_sgpa = float(request.form["curr_sgpa"])

        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        project = float(request.form["project"])

        reading = float(request.form["reading"])
        writing = float(request.form["writing"])

        # ML prediction (base)
        input_data = pd.DataFrame([{
            "gender": "male",
            "race/ethnicity": "group B",
            "parental level of education": "some college",
            "lunch": "standard",
            "test preparation course": "none",
            "reading score": reading,
            "writing score": writing
        }])

        score = model.predict(input_data)[0]

        # 🔥 REALISTIC IMPROVEMENT
        score += (prev_cgpa * 2)
        score += (prev_sgpa * 1.5)
        score += (curr_cgpa * 3)
        score += (curr_sgpa * 2)
        score += (study_hours * 1.2)
        score += (attendance * 0.2)
        score += (project * 0.3)

        score = max(0, min(score, 100))

        # Performance
        if score >= 75:
            perf = "Excellent"
        elif score >= 50:
            perf = "Average"
        else:
            perf = "Needs Improvement"

        # SAVE
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("INSERT INTO students (name, usn, branch, score, performance) VALUES (?, ?, ?, ?, ?)",
                (name, usn, branch, score, perf))
        conn.commit()
        conn.close()

        # GRAPH
        plt.figure()
        plt.bar(["Reading", "Writing", "Predicted"],
                [reading, writing, score])

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()

        return render_template("predict.html",
        prediction=perf,
        score=round(score, 2),
        graph_url=graph_url,
        name=name,
        usn=usn,
        branch=branch)

    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)