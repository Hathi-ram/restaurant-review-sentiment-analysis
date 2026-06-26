# 🍽 Restaurant Review Sentiment Analysis

A Machine Learning-based web application that analyzes restaurant reviews and predicts whether the sentiment is **Positive** or **Negative** using Natural Language Processing (NLP).

Built with

**Python**,

**Streamlit**,

**TF-IDF Vectorization**,

**Logistic Regression**.

---

## Project Overview

This project focuses on sentiment analysis of restaurant reviews.

The model processes customer reviews, cleans the text, converts it into numerical features using TF-IDF, and predicts sentiment.

This helps understand customer feedback automatically.

---
## <img width="2832" height="1514" alt="image" src="https://github.com/user-attachments/assets/382b6a53-d92b-4d37-a6ee-8df5fed7960a" />


## Features

Text preprocessing
Stopword removal
Stemming using PorterStemmer
TF-IDF vectorization
Sentiment prediction
Confidence score visualization
Sentiment probability pie chart
Top important words explanation
Prediction history
CSV export
Interactive Streamlit UI

---

## Models Tested

The following machine learning models were trained and compared:

* Naive Bayes
* Logistic Regression 
* Random Forest

### Best Model Selected: Logistic Regression

### Why Logistic Regression?

* Performs well with sparse TF-IDF vectors
* Provides probability scores
* More interpretable using coefficients
* Helps identify important positive/negative words
* Better generalization for text classification

---

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* Pandas
* Matplotlib
* NLTK
* Joblib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Hathi-ram/restaurant-review-sentiment-analysis.git
```

Go into the folder:

```bash
cd restaurant-review-sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

---

## Application Screenshots

### 1. Home Page

## <img width="2832" height="1514" alt="image" src="https://github.com/user-attachments/assets/382b6a53-d92b-4d37-a6ee-8df5fed7960a" />


---

### 2. Sentiment Prediction

## <img width="2824" height="1498" alt="image" src="https://github.com/user-attachments/assets/d10d1f02-a3f3-46fe-8301-7e9e7b230ef8" />



---

### 3. Confidence Score & Pie Chart

## <img width="2846" height="1502" alt="image" src="https://github.com/user-attachments/assets/d514d970-45ec-4685-9edb-e8b4cc921007" />



---

### 4. Top Important Words and  Prediction History

(Add screenshot here)


---
## <img width="2878" height="1502" alt="image" src="https://github.com/user-attachments/assets/ed3d45ee-8d17-458f-baf1-4a1d2a69b67c" />

---

## Workflow

1. Input restaurant review
2. Clean text using NLP
3. Apply stemming and stopword removal
4. Convert text into TF-IDF vectors
5. Predict sentiment using Logistic Regression
6. Display result with confidence score
7. Show important words and sentiment distribution

---

## Future Improvements

* Deploy on Streamlit Cloud
* Add BERT model
* Add LSTM model
* Multi-language support
* Real-time API integration

---

## Developer

**Vislavath Hathiram**
B.Tech Student
IIITDM Kancheepuram

 Chennai, India

### Connect with me:

GitHub: https://github.com/Hathi-ram
LinkedIn: https://www.linkedin.com/in/vislavath-hathiram-97a845358/
Email: [vislavathhathiram41@gmail.com](mailto:vislavathhathiram41@gmail.com)

---

## If you like this project

Give it a star on GitHub ⭐
