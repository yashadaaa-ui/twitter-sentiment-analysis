
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("C:\\Users\\litch\\Downloads\\twitter-sentiment-analysis\\data\\cleaned_tweets.csv")

df["cleaned_tweet"] = df["cleaned_tweet"].fillna("").astype(str)

X = df["cleaned_tweet"]
y = df["target"]

# Split before fitting TF-IDF
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fit only on training text
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

joblib.dump(
    (X_train, X_test, y_train, y_test),
    "models/train_test_data.pkl"
)

print("Train-test preparation completed!")
