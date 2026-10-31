import pandas as pd

df = pd.read_csv("C:\\Users\\litch\\Downloads\\twitter-sentiment-analysis\\data\\tweet_sentiment_10K.csv")
print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nSentiment Distribution:")
print(df.iloc[:, 0].value_counts())