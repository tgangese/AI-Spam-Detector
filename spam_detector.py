import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the SMS Spam dataset
df = pd.read_csv("spam.csv", sep="\t", header=None, names=["label", "message"], encoding="latin-1")
print(df.shape)
print(df.head())
print(df.columns)
print(df["label"].value_counts())
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates()
print("After dropping duplicates:", df.shape)
print(df["label"].value_counts())

def clean_text(text):
    text = text.lower()  # 1. lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # 2. keep only letters + spaces
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]  # 3. remove stopwords
    return " ".join(words)
# Test it on one message
print("Before:", df["message"].iloc[2])
print("After:", clean_text(df["message"].iloc[2]))
df["cleaned"] = df["message"].apply(clean_text)
print(df[["message", "cleaned"]].head(3))
X = df["cleaned"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", len(X_train), "Test:", len(X_test))
# TF-IDF: words -> numbers
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("Vocabulary size:", len(vectorizer.vocabulary_))
print("Train shape:", X_train_tfidf.shape)

# Train
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict on Test (exam)
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nReport:")
print(classification_report(y_test, y_pred))


def predict_email(text):
    cleaned_text = clean_text(text)
    vec = vectorizer.transform([cleaned_text])
    pred = model.predict(vec)[0]
    print(f"Email: {text}")
    print(f"→ Prediction: {pred.upper()}")
    print()

predict_email("Congratulations! You have won FREE money, click here to claim now!")
predict_email("Are we still meeting for lunch tomorrow at the office?")
predict_email("WINNER! Claim your prize urgently")