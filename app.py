
import streamlit as st
import pandas as pd
import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Page settings
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦"
)

st.title("🐦 Twitter Sentiment Analysis")
st.write("Analyze tweets using Machine Learning.")

# Prediction function
def predict_sentiment(text):
    transformed_text = tfidf.transform([text])
    prediction = model.predict(transformed_text)[0]

    if prediction == 1:
        return "Positive"
    else:
        return "Negative"


# Select analysis type
option = st.sidebar.selectbox(
    "Choose Analysis",
    ["Single Tweet", "Bulk CSV"]
)


# Single Tweet Analysis
if option == "Single Tweet":

    st.subheader("Analyze a Single Tweet")

    tweet = st.text_area("Enter your tweet:")

    if st.button("Analyze Sentiment"):

        if tweet.strip():

            result = predict_sentiment(tweet)

            if result == "Positive":
                st.success(f"Sentiment: {result}")
            else:
                st.error(f"Sentiment: {result}")

        else:
            st.warning("Please enter a tweet.")


# Bulk CSV Analysis
elif option == "Bulk CSV":

    st.subheader("Analyze Multiple Tweets")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("Uploaded Data:")
        st.dataframe(df)

        column = st.selectbox(
            "Select Tweet Column",
            df.columns
        )

        if st.button("Analyze All Tweets"):

            df["Predicted Sentiment"] = df[column].fillna("").astype(str).apply(
                predict_sentiment
            )

            st.subheader("Prediction Results")
            st.dataframe(df)

            st.subheader("Sentiment Distribution")
            st.bar_chart(
                df["Predicted Sentiment"].value_counts()
            )

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Results",
                data=csv,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )
