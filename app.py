import streamlit as st
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')
nltk.download('wordnet')


st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")

st.title("📊 Sentiment Analysis Web App")
st.write("Enter a review and the model will predict sentiment.")

## Load The Dataset

data = pd.read_csv("sentiment.csv")

## PREPROCESSING

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

data["clean_review"] = data["review"].apply(preprocess)

# TRAIN MODEL

X_train, X_test, y_train, y_test = train_test_split(
    data["clean_review"],
    data["sentiment"],
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# PREDICTION FUNCTION

def predict_sentiment(text):

    text = preprocess(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    if pred == 1:
        return "Positive 😊"
    else:
        return "Negative 😠"

# USER INPUT UI

review = st.text_area("✍ Enter your review here:")

if st.button("Predict Sentiment 🚀"):

    if review.strip() == "":
        st.warning("Please enter a review first!")

    else:
        result = predict_sentiment(review)
        if "Positive" in result:
            st.markdown(
                f"""
                <div style="background-color:#d4edda; color:#155724; padding:15px; 
                border-radius:10px; font-size:20px; font-weight:bold;">
                Prediction: {result}
                </div>
                """,
                unsafe_allow_html=True
                )
        else:
            st.markdown(
            f"""
            <div style="background-color:#f8d7da; color:#721c24; padding:15px; 
            border-radius:10px; font-size:20px; font-weight:bold;">
            Prediction: {result}
            </div>
            """,
            unsafe_allow_html=True
            )

# FOOTER

st.write("---")
st.write("Made with Maaz ❤️ using Streamlit")