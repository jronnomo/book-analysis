import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import streamlit as st
import plotly.express as px
from glob import glob
from pathlib import Path
import datetime

analyzer = SentimentIntensityAnalyzer()
english_stopwords = stopwords.words("english")

sentiments = []  # List of tuples containing (date, pos_sentiment, neg_sentiment)

filenames = glob('./diary/*txt')
for filename in filenames:
    with open(filename) as diary:
        entry = diary.read()
    analysis = analyzer.polarity_scores(entry)
    date = Path(filename).stem
    sentiments.append((date, analysis['pos'], analysis['neg']))

# Sort the sentiments list by date (earliest to latest)
sorted_sentiments = sorted(sentiments, key=lambda day_entry: datetime.datetime.strptime(day_entry[0], '%Y-%m-%d'))

# Extract the sorted dates and sentiments
sorted_dates = [item[0] for item in sorted_sentiments]
sorted_pos_sentiments = [item[1] for item in sorted_sentiments]
sorted_neg_sentiments = [item[2] for item in sorted_sentiments]

st.title("Diary Tone")
st.subheader("Positivity")
figure_pos = px.line(x=sorted_dates, y=sorted_pos_sentiments, labels={"x": "Date", "y": "Positivity"})
st.plotly_chart(figure_pos)
st.subheader("Negativity")
figure_neg = px.line(x=sorted_dates, y=sorted_neg_sentiments, labels={"x": "Date", "y": "Negativity"})
st.plotly_chart(figure_neg)
