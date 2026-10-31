
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLP resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load dataset
df = pd.read_csv("C:\\Users\\litch\\Downloads\\twitter-sentiment-analysis\\data\\cleaned_tweets.csv")

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_tweet(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags symbol but retain words
    text = re.sub(r"#", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenization
    words = text.split()

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Apply preprocessing
df["cleaned_tweet"] = df["tweet"].apply(clean_tweet)

# Save processed dataset
df.to_csv("data/cleaned_tweets.csv", index=False)

print("Preprocessing completed!")
print(df[["tweet", "cleaned_tweet"]].head())
