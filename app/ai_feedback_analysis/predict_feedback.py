import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import os

# Download NLTK data if not already present
nltk.download('punkt')
nltk.download('stopwords')

# Load saved vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load feedback data
df = pd.read_csv("feedback_sample.csv")
df.columns = df.columns.str.strip()  # Remove any whitespace from column names

# Generate botresponse column if missing or fill missing values
def generate_botresponse(row):
    if row['rating'].strip().lower() == 'positive':
        return "Thank you for your feedback! We're glad you found the information helpful."
    elif row['rating'].strip().lower() == 'negative':
        cat = row['category'].strip() if pd.notnull(row['category']) else ""
        if cat == 'Off-topic':
            return "Sorry about that! Let us know if you need assistance with your specific query."
        elif cat == 'Poor Phrasing':
            return "We apologize for the unclear explanation. Please let us know how we can improve it."
        elif cat == 'Too Technical':
            return "We understand that the explanation was complex. Let us know if you'd like more simplified details."
        elif cat == 'Not Helpful':
            return "Sorry that the information didn’t help. Feel free to ask for more details!"
        elif cat == 'Not Specific':
            return "We’ll be more specific in future responses. Let us know if you'd like further clarification."
        elif cat == 'Confusing':
            return "We apologize for the confusion. Could you clarify your query further?"
        elif cat == 'Irrelevant':
            return "We understand this wasn’t relevant. Let us know what you need help with."
        elif cat == 'Too Long':
            return "We’re sorry for the lengthy response. We'll keep it concise next time."
        elif cat == 'Not Clear':
            return "Sorry that the instructions weren't clear. We'll aim to simplify them."
        elif cat == 'Not Satisfactory':
            return "Sorry the response wasn't helpful. We’re improving based on your input."
        elif cat == 'Too Complicated':
            return "We understand the steps felt complex. Let us know if you'd like a simpler version."
        else:
            return "Thanks for your feedback. We'll strive to improve!"
    else:
        return "Thank you for your feedback!"

if 'botresponse' not in df.columns or df['botresponse'].isnull().any():
    df['botresponse'] = df.apply(generate_botresponse, axis=1)

# Filter only negative feedback
negative_df = df[df['rating'].str.lower().str.strip() == 'negative'].copy()

# Clean and vectorize comments
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    if pd.isnull(text):
        return ""
    words = word_tokenize(str(text).lower())
    words = [stemmer.stem(w) for w in words if w.isalpha() and w not in stop_words]
    return " ".join(words)

negative_df['clean_comment'] = negative_df['comment'].apply(clean_text)
comment_vectors = vectorizer.transform(negative_df['clean_comment'])

# Suggest fix function
def suggest_fix(new_comment):
    cleaned = clean_text(new_comment)
    new_vec = vectorizer.transform([cleaned])
    similarities = cosine_similarity(new_vec, comment_vectors)
    top_index = similarities.argmax()
    return negative_df.iloc[top_index]['botresponse']

# Test
if __name__ == "__main__":
    new_negative_input = "The answer was unclear and didn’t help."
    suggested_fix = suggest_fix(new_negative_input)
    print("Suggested Fix:", suggested_fix)