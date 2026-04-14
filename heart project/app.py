import json, joblib, os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Simple login system
VALID_USERS = {"admin": "admin123"}

# Load model + metadata
MODEL_PATH = os.path.join("artifacts", "model.pkl")
META_PATH = os.path.join("artifacts", "feature_metadata.json")

model = joblib.load(MODEL_PATH)
with open(META_PATH, "r") as f:
    metadata = json.load(f)

features = metadata["features"]
feature_info = metadata.get("feature_info", {})   # ✅ FIXED

@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("predict"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username in VALID_USERS and VALID_USERS[username] == password:
            session["user"] = username
            return redirect(url_for("predict"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session: 
        return redirect(url_for("login"))
    if request.method == "POST":
        try:
            data = [float(request.form[f]) for f in features]
            arr = np.array(data).reshape(1, -1)
            pred = model.predict(arr)[0]
            prob = model.predict_proba(arr)[0, 1] if hasattr(model, "predict_proba") else None
            return render_template("result.html", pred=pred, prob=prob)
        except Exception as e:
            return render_template("predict.html", features=features, feature_info=feature_info, error=str(e))
    return render_template("predict.html", features=features, feature_info=feature_info)

if __name__ == "__main__":
    app.run(debug=True)
