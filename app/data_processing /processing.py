import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime
import os
print(f"Current working directory: {os.getcwd()}")


nltk.download('punkt')
nltk.download('stopwords')
file_path = "../feedback.csv"

# Load CSV
df = pd.read_csv(file_path)
print("Columns in the CSV:", df.columns.tolist())
exit()

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Tokenize and clean comments
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text):
        return []
    words = word_tokenize(text.lower())
    return [w for w in words if w.isalpha() and w not in stop_words]

df['clean_tokens'] = df['comment'].apply(clean_text)

# Example: Count positive vs. negative
rating_counts = df['rating'].value_counts()

# Example: Top tokens in negative feedback
from collections import Counter
neg_tokens = df[df['rating'] == 'negative']['clean_tokens'].explode()
common_neg_words = Counter(neg_tokens).most_common(10)
common_positive_words = Counter(df[df['rating'] == 'positive']['clean_tokens'].explode()).most_common(1)
# Strip any leading/trailing spaces and convert to lowercase for consistency
df['rating'] = df['rating'].str.strip().str.lower()

# Check unique values in the 'rating' column to ensure consistency
print(df['rating'].unique())

# Check and print some rows from both categories to ensure data integrity
print(df[df['rating'] == 'positive'].head())
print(df[df['rating'] == 'negative'].head())

# Now perform the categorization with cleaned data
common_negative_categories = df[df['rating'] == 'negative']['category'].value_counts().head(5)
common_positive_categories = df[df['rating'] == 'positive']['category'].value_counts().head(5)

print("Top Negative Categories:\n", common_negative_categories)
print("Top Positive Categories:\n", common_positive_categories)

print("Top Negative Feedback Categories:\n", common_negative_categories)
print("Top Positive Feedback Categories:\n", common_positive_categories)
print("Rating Distribution:\n", rating_counts)
print("Top Negative Words:\n", common_neg_words)
print("Top Positive word:", common_positive_words)
