import pickle

# Load the trained model
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Example usage:
sample_text = "The support was very helpful and fast!"

# Clean the sample text (just like during training)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    if text is None:
        return ""
    words = word_tokenize(text.lower())
    words = [stemmer.stem(w) for w in words if w.isalpha() and w not in stop_words and len(w) > 2]
    return " ".join(words)

# Clean and vectorize the sample text
cleaned = clean_text(sample_text)
vectorized = vectorizer.transform([cleaned])

# Predict
prediction = model.predict(vectorized)

print("Prediction:", prediction[0])
