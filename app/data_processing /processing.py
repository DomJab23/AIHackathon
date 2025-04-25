import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime
import os
from collections import Counter

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# File path to your feedback CSV
file_path = "../feedback.csv"

# Load CSV
df = pd.read_csv(file_path)
print("Columns in the CSV:", df.columns.tolist())
exit()

# Check column names for consistency
print("Columns in the dataset:", df.columns)

# Convert timestamp to datetime, handle errors if timestamp column doesn't exist or is invalid
try:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
except Exception as e:
    print(f"Error converting timestamp: {e}")

# Tokenize and clean comments
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text):
        return []
    words = word_tokenize(text.lower())
    return [w for w in words if w.isalpha() and w not in stop_words]

df['clean_tokens'] = df['comment'].apply(clean_text)

# Check unique values in the 'rating' column
df['rating'] = df['rating'].str.strip().str.lower()
print("Unique ratings:", df['rating'].unique())

# Count positive vs. negative feedback
rating_counts = df['rating'].value_counts()

# Top tokens in negative feedback
neg_tokens = df[df['rating'] == 'negative']['clean_tokens'].explode()
common_neg_words = Counter(neg_tokens).most_common(10)

# Top tokens in positive feedback
common_positive_words = Counter(df[df['rating'] == 'positive']['clean_tokens'].explode()).most_common(1)

# Print some rows to verify data
print("Top Negative Categories:\n", df[df['rating'] == 'negative']['category'].value_counts().head(5))
print("Top Positive Categories:\n", df[df['rating'] == 'positive']['category'].value_counts().head(5))

# Output results
print("Rating Distribution:\n", rating_counts)
print("Top Negative Words:\n", common_neg_words)
print("Top Positive word:", common_positive_words)

# Ensure the dataframe looks correct
print(df.head())
