
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load dataset
df = pd.read_csv("C:\\Users\\litch\\Downloads\\twitter-sentiment-analysis\\data\\cleaned_tweets.csv")

print("Dataset Shape:", df.shape)

# Handle missing values
df["cleaned_tweet"] = df["cleaned_tweet"].fillna("")

# Sentiment distribution
print("\nSentiment Distribution:")
print(df["target"].value_counts())

# Calculate tweet length
df["tweet_length"] = df["cleaned_tweet"].apply(
    lambda x: len(str(x).split())
)

sns.set_style("whitegrid")

# 1. Sentiment Distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="target")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment (0 = Negative, 1 = Positive)")
plt.ylabel("Number of Tweets")
plt.savefig("data/sentiment_distribution.png")
plt.show()

# 2. Tweet Length Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["tweet_length"], bins=30, kde=True)
plt.title("Tweet Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.savefig("data/tweet_length.png")
plt.show()

# 3. Most Frequent Words
all_words = " ".join(df["cleaned_tweet"]).split()

word_counts = Counter(all_words).most_common(15)

if word_counts:
    words, counts = zip(*word_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words))
    plt.title("Top 15 Most Frequent Words")
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.tight_layout()
    plt.savefig("data/frequent_words.png")
    plt.show()

print("\nEDA completed successfully!")
