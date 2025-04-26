import pandas as pd
import nltk
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from gensim import corpora, models

# --- Setup ---
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
stopwords = nltk.corpus.stopwords.words("english")

# --- Load and Normalize Columns ---
df = pd.read_csv("feedback_sample.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
print("Available columns:", df.columns.tolist())

# Drop rows with missing comments
df = df.dropna(subset=["comment"])

# --- Recompute Clean Comments (if needed) ---
def preprocess(text):
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stopwords]
    return " ".join(tokens)

# Optional: Uncomment if clean_comment needs to be regenerated
# df["clean_comment"] = df["comment"].apply(preprocess)

# --- Visualization: Feedback Categories ---
plt.figure(figsize=(8, 5))
sns.countplot(data=df, y="category", order=df["category"].value_counts().index)
plt.title("Feedback Category Distribution")
plt.tight_layout()
plt.savefig("feedback_categories.png")
plt.show()

# --- Word Cloud of All Comments ---
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df["clean_comment"]))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Comments")
plt.savefig("wordcloud.png")
plt.show()

# --- Topic Modeling ---
texts = [comment.split() for comment in df["clean_comment"]]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

print("\nTop Topics Identified:")
for idx, topic in lda_model.print_topics(num_words=5):
    print(f"Topic {idx + 1}: {topic}")

# --- Backlog Suggestion Generation ---
category_counts = df["feedback_category"].value_counts()
backlog = []

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

for category, count in category_counts.items():
    suggestion = category_suggestions.get(category, "Review manually")
    backlog.append({
        "Category": category,
        "Occurrences": count,
        "Suggested Action": suggestion
    })

backlog_df = pd.DataFrame(backlog)
print("\nPrioritized Backlog Suggestions:")
print(backlog_df)

# --- Save Outputs ---
backlog_df.to_csv("ai_feedback_backlog.csv", index=False)
df.to_csv("classified_feedback.csv", index=False)
from topic_extraction import get_conversation_topic

backlog = []

for feedback in feedback_list:
    if feedback['category'] is None and feedback['comment'] is None:
        # This feedback only had thumbs up/down, no category or comment
        conversation_text = feedback['conversation']  # You must have stored the chat
        topic = get_conversation_topic(conversation_text)

        backlog.append({
            "Category": "UNSPECIFIED",
            "Occurrences": 1,  # or however you are counting
            "Suggested_Action": "Review manually",
            "Topics": [topic]
        })
    else:
        # Normal handling for other feedbacks
        backlog.append({
            "Category": feedback['category'],
            "Occurrences": feedback['count'],
            "Suggested_Action": "Review manually"
        })
