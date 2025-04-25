from flask import Flask, render_template, request, redirect
import uuid
import csv
import os
from datetime import datetime

app = Flask(__name__)

@app.route("/", methods=["GET"])
def feedback_form():
    user_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    return render_template("index.html", user_id=user_id, session_id=session_id)

@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    data = {
        "user_id": request.form.get("user_id"),
        "session_id": request.form.get("session_id"),
        "rating": request.form.get("rating"),
        "category": request.form.get("category"),
        "comment": request.form.get("comment"),
        "related_query": request.form.get("related_query"),
        "timestamp": datetime.now().isoformat()
    }

    file_exists = os.path.isfile("feedback.csv")
    with open("feedback.csv", mode="a", newline="") as csvfile:
        fieldnames = data.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

    return redirect("/")
if __name__ == "__main__":
    app.run(debug=True, port=5001)

