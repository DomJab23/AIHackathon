import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import os

nltk.download('punkt')
nltk.download('stopwords')

file_path = "feedback.csv"

extra_stopwords = {'ok', 'okay', 'sure', 'yeah', 'uh', 'hmm', 'hi', 'hello', 'thanks', 'thank'}
stop_words = set(stopwords.words('english')).union(extra_stopwords)


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text):
        return []

    words = word_tokenize(text.lower())
    tagged_words = pos_tag(words)

    filtered_phrases = []
    i = 0
    while i < len(tagged_words):
        word, tag = tagged_words[i]

        # Pomijamy niealfabetyczne słowa i stopwords
        if not word.isalpha() or word in stop_words:
            i += 1
            continue

        phrase = [word]

        # Próbujemy stworzyć 2- lub 3-wyrazowe frazy
        if i + 1 < len(tagged_words):
            next_word, next_tag = tagged_words[i+1]
            if next_word.isalpha() and next_word not in stop_words:
                # 2-wyrazowa fraza
                phrase.append(next_word)

                # Sprawdźmy czy można jeszcze dodać trzecie słowo
                if i + 2 < len(tagged_words):
                    third_word, third_tag = tagged_words[i+2]
                    if third_word.isalpha() and third_word not in stop_words:
                        # 3-wyrazowa fraza
                        phrase.append(third_word)
                        filtered_phrases.append('_'.join(phrase))
                        i += 3
                        continue

                filtered_phrases.append('_'.join(phrase))
                i += 2
                continue

        # Jeśli nie udało się połączyć, dodaj pojedyncze znaczące słowo
        filtered_phrases.append(word)
        i += 1

    return filtered_phrases


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
