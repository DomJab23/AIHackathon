import sys
import os
import csv
from datetime import datetime
from flask import Flask, render_template, request, redirect, session
import uuid
import locale
import requests
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from data_processing.processing import get_feedback_analysis

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a real secret key

# Load vectorizer
with open("ai_feedback_analysis/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load feedback sample for suggestion system
feedback_sample_df = pd.read_csv("ai_feedback_analysis/feedback_sample.csv")

# Setup NLTK
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    if pd.isnull(text):
        return ""
    words = word_tokenize(text.lower())
    words = [stemmer.stem(w) for w in words if w.isalpha() and w not in stop_words]
    return " ".join(words)

# Prepare negative feedback
negative_df = feedback_sample_df[feedback_sample_df['rating'].str.lower().str.strip() == 'negative'].copy()
negative_df['clean_comment'] = negative_df['comment'].apply(clean_text)
comment_vectors = vectorizer.transform(negative_df['clean_comment'])

def suggest_fix(new_comment):
    cleaned = clean_text(new_comment)
    new_vec = vectorizer.transform([cleaned])
    similarities = cosine_similarity(new_vec, comment_vectors)
    top_index = similarities.argmax()
    return negative_df.iloc[top_index]['botresponse']

# Get user country
def get_user_country():
    try:
        response = requests.get("https://ipinfo.io/json")  # More stable version
        if response.status_code == 200:
            location_data = response.json()
            return location_data.get('country', 'Unknown')
    except Exception as e:
        print(f"Error fetching country: {e}")
    return "Unknown"

# Define category suggestions
category_suggestions = {
    "Incorrect Answer": "Improve intent recognition and response matching",
    "Missing Knowledge": "Expand knowledge base for common questions",
    "Poor Phrasing": "Improve language generation templates for clarity",
    "Too Technical": "Simplify technical language for general users",
    "Not Specific Enough": "Provide more detailed and task-specific responses",
    "Off-topic": "Improve contextual relevance of answers",
    "Technical Issue": "Investigate performance/stability of assistant backend",
    "Helpful": "Maintain strengths in response clarity and usefulness",
    "Excellent Service": "Encourage consistency in high-quality support",
    "Thank You": "Leverage positive signals for training data",
    "Other": "Review manually for unique cases"
}

# Analytics route
@app.route("/analytics")
def show_analytics():
    analysis = get_feedback_analysis()

    backlog = []
    category_counts = analysis.get('category_counts', {})
    for category, count in category_counts.items():
        suggestion = category_suggestions.get(category, "Review manually")
        backlog.append({
            "Category": category,  # Uppercase first letter to match template
            "Occurrences": count,
            "Suggested_Action": suggestion
        })
    
    return render_template("analytics.html", data=analysis, backlog=backlog)

@app.route("/", methods=["GET"])
def quick_feedback():
    return render_template("simple_feedback.html")

# Feedback form route
@app.route("/index.html", methods=["GET"])
def feedback_form():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    session['session_id'] = str(uuid.uuid4())
    
    return render_template("index.html", user_id=session['user_id'], session_id=session['session_id'])


# Submit feedback route
@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    session['session_id'] = str(uuid.uuid4())
    current_locale, _ = locale.getdefaultlocale()
    user_country = get_user_country()

    rating = request.form.get("rating")
    comment = request.form.get("comment")
    category = request.form.get("category") or "Unspecified"

    data = {
        "user_id": request.form.get("user_id"),
        "session_id": session['session_id'],
        "rating": rating,
        "category": category,
        "comment": comment,
        "related_query": request.form.get("related_query"),
        "timestamp": datetime.now().isoformat(),
        "locale": current_locale,
        "country": user_country,
    }

    # Save to feedback.csv
    fieldnames = ["user_id", "session_id", "rating", "category", "comment", "related_query", "timestamp", "locale", "country"]
    file_exists = os.path.isfile("feedback.csv")
    is_empty = os.stat("feedback.csv").st_size == 0 if file_exists else True

    with open("feedback.csv", mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if is_empty:
            writer.writeheader()
        writer.writerow(data)

    # Generate response
    if rating == "negative":
        bot_response = suggest_fix(comment)
    else:
        bot_response = "Thank you for your feedback! We're glad you found the information helpful."

    return render_template("response.html", bot_response=bot_response)

# Run app
if __name__ == "__main__":
    app.run(debug=True, port=5001)
