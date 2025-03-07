import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from .predictions import predict_tweet_sentiment

import streamlit as st
from collections import Counter


def generate_wordcloud(texts):
    wordcloud = WordCloud(max_words=100, width=800, height=400).generate(" ".join(texts))

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    return fig


def plot_top_orgs(df_counts, top_n):

    df_top_counts = df_counts.head(top_n)

    fig = px.bar(df_top_counts, x="Value", y="Count", title=f"Top {top_n} Organisations par Count",
                 labels={"Value": "Organisation", "Count": "Nombre d'occurrences"},
                 text_auto=True)

    return df_top_counts, fig

def plot_pie_chart(data):
    counts = data.value_counts().reset_index()
    counts.columns = ['Sentiment', 'Count']

    fig = px.pie(
        counts,
        names='Sentiment',
        values='Count',
        title="Category Sentiment",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hole=0.3
    )

    fig.update_traces(textinfo='percent+label', pull=[0.1, 0, 0])
    return fig

def plot_top_n_sentiment_multibar(df, top_n):

    sentiment_counts = df.groupby("bert_orgs")["polarity_predictions"].value_counts().unstack().fillna(0)

    sentiment_percent = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100

    top_n_sentiment_counts = sentiment_counts.sort_values("positive", ascending=False).head(top_n)
    top_n_sentiment_percent = sentiment_percent.loc[top_n_sentiment_counts.index]

    traces = []
    for sentiment in top_n_sentiment_counts.columns:
        trace = go.Bar(
            x=top_n_sentiment_counts.index,
            y=top_n_sentiment_counts[sentiment],
            name=sentiment,
            hoverinfo="x+y+text",
            text=top_n_sentiment_percent[sentiment].round(1).astype(str) + "%",
            textposition='inside',
            marker=dict(color={"negative": "red", "neutral": "gray", "positive": "green"}[sentiment]),
            offsetgroup=sentiment
        )
        traces.append(trace)

    layout = go.Layout(
        title=f"Top {top_n} Companies by Count of Positive Sentiment Tweets",
        xaxis=dict(title="Company"),
        yaxis=dict(title="Number of Tweets"),
        barmode="stack",
        bargap=0.1,
        hovermode="x unified",
        legend_title="Sentiment",
        template="plotly_white",
        xaxis_tickangle=-45,
    )

    fig = go.Figure(data=traces, layout=layout)

    return fig

def process_top_organizations(df, top_n=5):
    df = df.explode("bert_orgs").dropna(subset=["bert_orgs"])
    value_counts = Counter(df["bert_orgs"])
    df_counts = pd.DataFrame(value_counts.items(), columns=["Value", "Count"]).sort_values(by="Count", ascending=False)

    df_top, fig = plot_top_orgs(df_counts, top_n=top_n)
    return df_top, fig

def get_visualization(df, visualize_option, top_n=5):
    if visualize_option == "Word Cloud":
        fig = generate_wordcloud(df["processed_text"])
        return  None, fig

    elif visualize_option == "Bar Plot":
        df_top, fig = process_top_organizations(df, top_n)
        return df_top, fig

    return None, None

def display_predictions(tweet_df):
    tweet_df = predict_tweet_sentiment(tweet_df)

    fig = plot_pie_chart(tweet_df["polarity_predictions"])
    st.plotly_chart(fig)

    df = tweet_df.explode("bert_orgs").dropna(subset=["bert_orgs"])
    top_n = st.sidebar.slider("Number of top organizations to display", min_value=1, max_value=10, value=5)
    st.plotly_chart(plot_top_n_sentiment_multibar(df, top_n))

def get_predictions_and_figures(tweet_df, top_n=5):
    tweet_df = predict_tweet_sentiment(tweet_df)

    pie_chart_fig = plot_pie_chart(tweet_df["polarity_predictions"])

    df = tweet_df.explode("bert_orgs").dropna(subset=["bert_orgs"])
    multibar_fig = plot_top_n_sentiment_multibar(df, top_n)

    return pie_chart_fig, multibar_fig