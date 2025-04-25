import os
import csv
from datetime import datetime
from flask import Flask, render_template, request, redirect, session
import uuid
import locale
import requests

app = Flask(__name__)

# Set a secret key for session management
app.secret_key = 'your_secret_key_here'  # Replace with your own secret key

# Function to get country based on user's IP
def get_user_country():
    try:
        # Send a request to ipinfo.io to get location info based on IP
        response = requests.get("http://ipinfo.io")
        location_data = response.json()

        # Extract country from the location data
        country = location_data.get('country', 'Unknown')
        return country
    except Exception as e:
        print(f"Error fetching country: {e}")
        return "Unknown"

@app.route("/", methods=["GET"])
def feedback_form():
    # Check if the session already has user_id
    if 'user_id' not in session:
        # If not, generate and store new user_id in the session
        session['user_id'] = str(uuid.uuid4())
    
    # Retrieve user_id from the session
    user_id = session['user_id']
    
    # Generate a new session_id on each form load (so it's unique every time)
    session['session_id'] = str(uuid.uuid4())
    
    return render_template("index.html", user_id=user_id, session_id=session['session_id'])

@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    # Generate a new session_id for each submission
    session['session_id'] = str(uuid.uuid4())
    
    # Get system locale (you can use this wherever needed in the app)
    current_locale, encoding = locale.getdefaultlocale()
    
    # Fetch the country based on user's IP address
    user_country = get_user_country()

    data = {
        "user_id": request.form.get("user_id"),
        "session_id": session['session_id'],  # Use the updated session_id
        "rating": request.form.get("rating"),
        "category": request.form.get("category"),
        "comment": request.form.get("comment"),
        "related_query": request.form.get("related_query"),
        "timestamp": datetime.now().isoformat(),
        "locale": current_locale,  # Add system locale to the feedback data
        "country": user_country,   # Add the user's country based on IP
    }

    # Define the header names
    fieldnames = ["user_id", "session_id", "rating", "category", "comment", "related_query", "timestamp", "locale", "country"]

    # Check if the file exists and is empty (first time writing)
    file_exists = os.path.isfile("feedback.csv")
    is_empty = os.stat("feedback.csv").st_size == 0 if file_exists else True

    # Open the CSV file to write data
    with open("feedback.csv", mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header only if the file is empty or new
        if is_empty:
            writer.writeheader()
        
        # Write the data row
        writer.writerow(data)

    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
