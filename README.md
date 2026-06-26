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
## <img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/338a72d8-ab06-46d3-b907-d544afea338c" />

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

(Add screenshot here)

![Home Page](images/home.png)

---

### 2. Sentiment Prediction

(Add screenshot here)

![Prediction Result](images/prediction.png)

---

### 3. Confidence Score & Pie Chart

(Add screenshot here)

![Confidence Score](images/confidence.png)

---

### 4. Top Important Words

(Add screenshot here)

![Important Words](images/important_words.png)

---

### 5. Prediction History

(Add screenshot here)

![Prediction History](images/history.png)

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
