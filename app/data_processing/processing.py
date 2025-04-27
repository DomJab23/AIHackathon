import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import os
from nltk import pos_tag

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

file_path = "feedback.csv"

extra_stopwords = {'ok', 'okay', 'sure', 'yeah', 'uh', 'hmm', 'hi', 'hello', 'thanks', 'thank',
                   'okay', 'sure', 'yeah', 'hmm', 'hello', 'thanks', 'thank', 'yup', 'nah',
    'hmm', 'huh', 'lol', 'lmao', 'okay',
    'yes', 'maybe', 'idk', 'hahaha', 'haha', 'heh', 'meh',
    'huh', 'hmmm', 'hmm', 'huhuh', 'yup', 'nope', 'yeah',
    'hahah', 'uhh', 'umm', 'omg', 'wtf', 'smh', 'hahaha',
    'hello', 'yo', 'sup', 'hey', 'thx',
    'pls', 'plz', 'k', 'kk', 'brb', 'gtg', 'btw', 'idc',
    'ikr', 'tbh', 'bff', 'rofl'
    'i'}
stop_words = set(stopwords.words('english')).union(extra_stopwords)

def is_garbage_word(word):
    if len(word) < 3:
        return True
    if all(c == word[0] for c in word):  # aaa, bbb, etc.
        return True
    if word in extra_stopwords:
        return True
    return False

def clean_text(text):
    if pd.isnull(text):
        return []

    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    
    important_single = {
        'not', 'no', 'never', 'cannot', "can't", "didn't", "won't", "shouldn't", 
        "wouldn't", "couldn't", 'many', 'few', 'some', 'several', 'lot', 'lots', 
        'plenty', 'little', 'much', 'too', 'wrong'
    }

    important_phrases = [
        ('a', 'lot', 'of'),
        ('lot', 'of'),
        ('plenty', 'of'),
        ('a', 'few'),
        ('not','what','i')
    ]

    filtered_words = []
    i = 0
    while i < len(words):
        word = words[i]

        # Sprawdzanie czy aktualne słowa tworzą ważną frazę
        matched_phrase = False
        for phrase in important_phrases:
            if words[i:i+len(phrase)] == list(phrase):
                # Łączymy frazę (np. a + lot + of + next_word)
                next_word_idx = i + len(phrase)
                if next_word_idx < len(words) and words[next_word_idx].isalpha():
                    combined = f"{phrase[-1]}_{words[next_word_idx]}"
                    filtered_words.append(combined)
                    i = next_word_idx + 1
                    matched_phrase = True
                    break

        if matched_phrase:
            continue

        # Łączenie pojedynczych ważnych słów z następnym słowem
        if word in important_single and i + 1 < len(words) and words[i+1].isalpha():
            combined = f"{word}_{words[i+1]}"
            filtered_words.append(combined)
            i += 2
            continue

        # Normalne pojedyncze słowa
        if word.isalpha() and word not in stop_words and not is_garbage_word(word):
            filtered_words.append(word)

        i += 1

    return filtered_words



def get_feedback_analysis():
    # Wczytanie danych feedbacku
    if not os.path.exists(file_path):
        return {}

    df = pd.read_csv(file_path)

    # Podstawowa obróbka danych
    df['rating'] = df['rating'].str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['clean_tokens'] = df['comment'].apply(clean_text)

    # Analiza ocen
    rating_counts = df['rating'].value_counts().to_dict()

    # Analiza kategorii
    common_negative_categories = df[df['rating'] == 'negative']['category'].value_counts().to_dict()
    common_positive_categories = df[df['rating'] == 'positive']['category'].value_counts().to_dict()

    # Najczęstsze słowa w feedbacku
    neg_tokens = df[df['rating'] == 'negative']['clean_tokens'].explode()
    common_neg_words = Counter(neg_tokens).most_common(10)

    pos_tokens = df[df['rating'] == 'positive']['clean_tokens'].explode()
    common_positive_words = Counter(pos_tokens).most_common(5)

    # Rozkład krajów
    top_country = df['country'].value_counts().head(5).to_dict()
    all_countries = df['country'].value_counts().to_dict()

    # Nowość: Analiza kategorii dla wszystkich feedbacków
    category_counts = df['category'].value_counts().to_dict()

    user_feedback_counts = Counter(df['user_id'])
    top_users = user_feedback_counts.most_common(10)

    return {
        'rating_counts': rating_counts,
        'common_negative_categories': common_negative_categories,
        'common_positive_categories': common_positive_categories,
        'common_neg_words': common_neg_words,
        'common_positive_words': common_positive_words,
        'top_country': top_country,
        'all_countries': all_countries,
        'category_counts': category_counts,
        'top_users': top_users,
    }
