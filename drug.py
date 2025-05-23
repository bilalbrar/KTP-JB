# -*- coding: utf-8 -*-


# Imports
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load datasets
data_path_train = r"C:\Users\bilal\BCU\Drug Reviews Dataset\drugsComTrain_raw.csv"
data_path_test  = r"C:\Users\bilal\BCU\Drug Reviews Dataset\drugsComTest_raw.csv"
df_train = pd.read_csv(data_path_train)[['review','drugName','condition','rating']]
df_test  = pd.read_csv(data_path_test)[['review','drugName','condition','rating']]

# Map ratings to sentiment
def map_sentiment(r):
    if r <= 4:
        return 'negative'
    elif r <= 6:
        return 'neutral'
    else:
        return 'positive'

df_train['sentiment'] = df_train['rating'].apply(map_sentiment)
df_test['sentiment']  = df_test ['rating'].apply(map_sentiment)

# Preprocess text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df_train['cleaned_review'] = df_train['review'].astype(str).apply(clean_text)
df_test ['cleaned_review'] = df_test ['review'].astype(str).apply(clean_text)

# Prepare train/test sets (no train_test_split)
X_train, y_train = df_train['cleaned_review'], df_train['sentiment']
X_test,  y_test  = df_test ['cleaned_review'], df_test ['sentiment']

# Define pipelines
pipelines = {
    'logreg': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20_000, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'nb': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20_000, ngram_range=(1,2))),
        ('clf', MultinomialNB())
    ]),
    'svm': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20_000, ngram_range=(1,2))),
        ('clf', LinearSVC())
    ]),
}

# Train models
for name, pipe in pipelines.items():
    print(f"Training {name}...")
    pipe.fit(X_train, y_train)

# Evaluate and select best model
best_model_name = None
best_f1_macro = 0

print("\n=== MODEL EVALUATION ===")
for name, pipe in pipelines.items():
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"\n{name.upper()}: Acc={acc:.4f}, Macro-F1={f1_macro:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred))
    if f1_macro > best_f1_macro:
        best_f1_macro = f1_macro
        best_model_name = name

# Save best model
os.makedirs('models', exist_ok=True)
joblib.dump(pipelines[best_model_name], 'models/best_sentiment_model.pkl')
print(f"Saved best model ({best_model_name}) with Macro-F1={best_f1_macro:.4f}")



