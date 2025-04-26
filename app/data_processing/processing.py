import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import os

nltk.download('punkt')
nltk.download('stopwords')

file_path = "feedback.csv"

stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text):
        return []
    words = word_tokenize(text.lower())
    return [w for w in words if w.isalpha() and w not in stop_words]

def get_feedback_analysis():
    # Load latest feedback
    if not os.path.exists(file_path):
        return {}

    df = pd.read_csv(file_path)

    # Basic cleanup
    df['rating'] = df['rating'].str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['clean_tokens'] = df['comment'].apply(clean_text)

    # Ratings breakdown
    rating_counts = df['rating'].value_counts().to_dict()

    # Category breakdown
    common_negative_categories = df[df['rating'] == 'negative']['category'].value_counts().to_dict()
    common_positive_categories = df[df['rating'] == 'positive']['category'].value_counts().to_dict()

    # Common words in feedback
    neg_tokens = df[df['rating'] == 'negative']['clean_tokens'].explode()
    common_neg_words = Counter(neg_tokens).most_common(10)

    pos_tokens = df[df['rating'] == 'positive']['clean_tokens'].explode()
    common_positive_words = Counter(pos_tokens).most_common(5)

    # Country breakdown
    top_country = df['country'].value_counts().head(5).to_dict()
    all_countries = df['country'].value_counts().to_dict()

    # NEW: Category counts for all feedback (used for backlog)
    category_counts = df['category'].value_counts().to_dict()

    return {
        'rating_counts': rating_counts,
        'common_negative_categories': common_negative_categories,
        'common_positive_categories': common_positive_categories,
        'common_neg_words': common_neg_words,
        'common_positive_words': common_positive_words,
        'top_country': top_country,
        'all_countries': all_countries,
        'category_counts': category_counts,  # <-- added to fix analytics page
    }
