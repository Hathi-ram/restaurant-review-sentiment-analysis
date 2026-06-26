import streamlit as st
import joblib
import os
import re
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords
nltk.download("stopwords")

# Load model and vectorizer
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "best_sentiment_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

model = joblib.load(model_path)
cv = joblib.load(vectorizer_path)

# Stemmer
ps = PorterStemmer()

# Custom stopwords
custom_stopwords = {
    'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
    'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'no', 'nor',
    'not', 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

stop_words = set(stopwords.words("english")) - custom_stopwords


# Clean text function
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)


# Page config
st.set_page_config(
    page_title="Restaurant Review Analyzer",
    page_icon="🍽",
    layout="centered"
)

# ================= Sidebar =================
st.sidebar.title("👨‍💻 Developer Info")

# Developer details
st.sidebar.markdown("### Vislavath Hathiram")
st.sidebar.write(" B.Tech Student")
st.sidebar.write(" IIITDM Kancheepuram")
st.sidebar.write(" AI | Machine Learning | NLP")
st.sidebar.write(" Chennai, India")

st.sidebar.markdown("---")

# Project info
st.sidebar.subheader("Project Info")
st.sidebar.write("🍽 Restaurant Review Sentiment Analysis")

st.sidebar.markdown("### Models Tested")
st.sidebar.write("• Naive Bayes")
st.sidebar.write("• Logistic Regression")
st.sidebar.write("• Random Forest")

st.sidebar.markdown("### Best Model Selected")
st.sidebar.write("Logistic Regression")

st.sidebar.markdown("###  Why Logistic Regression?")
st.sidebar.write("""
• Better for sparse TF-IDF vectors  
• Gives probability scores for confidence  
• More interpretable using word coefficients  
• Helps identify important positive/negative words  
• Better generalization for text classification  
""")

st.sidebar.write("Vectorizer: TF-IDF")

st.sidebar.markdown("---")

# Contact links
st.sidebar.subheader(" Connect")
st.sidebar.markdown(" GitHub: [Hathi-ram](https://github.com/Hathi-ram)")
st.sidebar.markdown(" LinkedIn: [Vislavath Hathiram](https://www.linkedin.com/in/vislavath-hathiram-97a845358/)")
st.sidebar.markdown(" Email: vislavathhathiram41@gmail.com")

st.sidebar.markdown("---")
st.sidebar.write("Loaded Model:", type(model).__name__)

# ================= Main UI =================
st.title("🍽 Restaurant Review Sentiment Analysis")
st.write("Analyze restaurant reviews instantly using Machine Learning.")

# Sample buttons
st.subheader("Try Sample Reviews")

col1, col2 = st.columns(2)

if col1.button("Positive Example"):
    st.session_state.review = "The food was amazing and service was excellent"

if col2.button("Negative Example"):
    st.session_state.review = "The food was terrible and not tasty"

# Input box
review = st.text_area(
    "Write your review here:",
    value=st.session_state.get("review", "")
)

# Predict button
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review first.")

    else:
        cleaned_review = clean_text(review)

        vector = cv.transform([cleaned_review]).toarray()

        prediction = model.predict(vector)
        probability = model.predict_proba(vector)

        confidence = max(probability[0]) * 100

        # Review stats
        st.subheader(" Review Statistics")

        col1, col2 = st.columns(2)
        col1.metric("Words", len(review.split()))
        col2.metric("Characters", len(review))

        # Result
        st.subheader(" Prediction Result")

        if prediction[0] == 1:
            st.success("Positive Review ")
            st.balloons()
        else:
            st.error("Negative Review ")

        # Confidence
        st.subheader(" Confidence Score")
        st.progress(int(confidence))
        st.metric("Confidence", f"{confidence:.2f}%")

        # Pie chart
        st.subheader(" Sentiment Distribution")

        labels = ["Negative", "Positive"]
        sizes = probability[0]

        fig, ax = plt.subplots()
        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )

        st.pyplot(fig)

        # Top important words
        st.subheader(" Top Important Words")

        feature_names = cv.get_feature_names_out()
        coefficients = model.coef_[0]

        important_words = sorted(
            zip(feature_names, coefficients),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]

        important_df = pd.DataFrame(
            important_words,
            columns=["Word", "Impact Score"]
        )

        st.dataframe(important_df)

        # Save history
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Review": review,
            "Prediction": "Positive" if prediction[0] == 1 else "Negative",
            "Confidence": round(confidence, 2)
        })

# History section
if "history" in st.session_state and len(st.session_state.history) > 0:
    st.subheader(" Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df)

    # Download CSV
    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

# Clear history
if st.button("🗑 Clear History"):
    st.session_state.history = []
    st.success("History cleared successfully.")

# Footer
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ by Vislavath Hathiram using Streamlit</center>",
    unsafe_allow_html=True
)
