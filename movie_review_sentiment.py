from flask import Flask, render_template, request
import pandas as pd
import re
import os 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle


app = Flask(__name__)

# Load the DataFrame from pickle file
df = pickle.load(open('movies.pkl', 'rb'))

# Load the dictionary from pickle file
df_dict = pickle.load(open('movies_dictionary.pkl', 'rb'))

# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# Define data processing function
def data_processing(text):
    text= text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

# Apply data processing to the dataset
df['review'] = df['review'].apply(data_processing)

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(preprocessor=data_processing)
X_tfidf = tfidf_vectorizer.fit_transform(df['review'])

# Train logistic regression model
logreg = LogisticRegression()
logreg.fit(X_tfidf, df['sentiment'])

# Function to predict sentiment of the review
def predict_sentiment(review):
    review_tfidf = tfidf_vectorizer.transform([review])
    return logreg.predict(review_tfidf)[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        sentiment = predict_sentiment(review)
        return render_template('result.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
