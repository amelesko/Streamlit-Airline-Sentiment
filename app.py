'''
An app to show airline tweet sentiment.

Created as part of the Coursera project "Create Interactive Dashboards
    with Streamlit and Python".

Created by: Alex Melesko
Date: 1/12/2022
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS

# Define global variables first
DATA_URL = (
    "Tweets.csv"
)

# Set up titles
st.title("Sentiment Analysis of Tweets about US Airlines")
st.sidebar.title("Sentiment Analysis of Tweets")
st.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets 🐦")
st.sidebar.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets 🐦")

# Loads specific dataset and caches
@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()

# Add subsection to show a random tweet based on sentiment type
st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
# Chooses one tweet that matches the sentiment chosen from the radio
st.sidebar.markdown(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0])

# Add a subsection to viz the amount of tweets by sentiment type
st.sidebar.markdown("### Number of tweets by sentiment")
select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='sentiment_count_selectbox')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
# Ask user if they want to show charts
if not st.sidebar.checkbox("Hide", True, key='show_tweet_count_checkbox'):
    st.markdown("### Number of tweets by sentiment")
    # Show either bar chart or pie chart
    if select == 'Bar plot':
        fig_sentiment_count = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig_sentiment_count)
    else:
        fig_sentiment_count = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig_sentiment_count)

# Viz the tweets by hour of the day
st.sidebar.subheader("When and where are users tweeting from?")
hour = st.sidebar.slider("Hour to look at", 0, 23)
modified_data = data[data['tweet_created'].dt.hour == hour]
if not st.sidebar.checkbox("Close", True, key='show_tweet_times_checkbox'):
    st.markdown("### Tweet locations based on time of day")
    st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour + 1) % 24))
    st.map(modified_data)

# Viz tweet count by airline
st.sidebar.subheader("Total number of tweets for each airline")
each_airline = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='airline_count_selectbox')
airline_sentiment_count = data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
airline_sentiment_count = pd.DataFrame({'Airline':airline_sentiment_count.index, 'Tweets':airline_sentiment_count.values.flatten()})
# Ask user if they want to show charts
if not st.sidebar.checkbox("Close", True, key='show_tweet_charts_checkbox'):
    # Show either bar chart or pie chart
    if each_airline == 'Bar plot':
        st.subheader("Total number of tweets for each airline")
        fig_airline_count = px.bar(airline_sentiment_count, x='Airline', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig_airline_count)
    if each_airline == 'Pie chart':
        st.subheader("Total number of tweets for each airline")
        fig_airline_count = px.pie(airline_sentiment_count, values='Tweets', names='Airline')
        st.plotly_chart(fig_airline_count)

# Plots specific sentiment data and caches
@st.cache(persist=True)
def plot_sentiment(airline):
    df = data[data['airline']==airline]
    count = df['airline_sentiment'].value_counts()
    count = pd.DataFrame({'Sentiment':count.index, 'Tweets':count.values.flatten()})
    return count

# Viz tweet sentiment by airline
st.sidebar.subheader("Breakdown sentiment by airline")
choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'), key='sentiment_airline_multiselect')
if len(choice) > 0:
    st.subheader("Breakdown airline by sentiment")
    breakdown_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot', ], key='airline_sentiment_count_selectbox')
    fig_airline_sentiment_count = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
    if breakdown_type == 'Bar plot':
        for i in range(1):
            for j in range(len(choice)):
                fig_airline_sentiment_count.add_trace(
                    go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).Tweets, showlegend=False),
                    row=i+1, col=j+1
                )
        fig_airline_sentiment_count.update_layout(height=600, width=800)
        st.plotly_chart(fig_airline_sentiment_count)
    else:
        fig_airline_sentiment_count = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
        for i in range(1):
            for j in range(len(choice)):
                fig_airline_sentiment_count.add_trace(
                    go.Pie(labels=plot_sentiment(choice[j]).Sentiment, values=plot_sentiment(choice[j]).Tweets, showlegend=True),
                    i+1, j+1
                )
        fig_airline_sentiment_count.update_layout(height=600, width=800)
        st.plotly_chart(fig_airline_sentiment_count)
        # Viz histogram of sentiment by airline
st.sidebar.subheader("Breakdown airline by sentiment")
choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'), key='sentiment_airline_histogram_multiselect')
if len(choice) > 0:
    choice_data = data[data.airline.isin(choice)]
    histogram_airline_sentiment = px.histogram(
                        choice_data, x='airline', y='airline_sentiment',
                         histfunc='count', color='airline_sentiment',
                         facet_col='airline_sentiment', labels={'airline_sentiment':'tweets'},
                          height=600, width=800)
    st.plotly_chart(histogram_airline_sentiment)

# Create word cloud based on sentiment type
st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
if not st.sidebar.checkbox("Close", True, key='show_wordcloud_checkbox'):
    st.subheader('Word cloud for %s sentiment' % (word_sentiment))
    df = data[data['airline_sentiment']==word_sentiment]
    words = ' '.join(df['text'])
    # Remove any known issue words "http", startswith @ and "RT"
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()
    
# Show raw data if the user likes
st.sidebar.header("Raw Data")
if st.sidebar.checkbox("Show raw data", False, key='show_raw_data_checkbox'):
    st.write(data)
