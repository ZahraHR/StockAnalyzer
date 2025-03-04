import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from .predictions import process_tweet_data

import streamlit as st
from collections import Counter


def generate_wordcloud(texts):
    wordcloud = WordCloud(max_words=100, width=800, height=400).generate(" ".join(texts))

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)


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

    top5_sentiment_counts = sentiment_counts.sort_values("positive", ascending=False).head(top_n)
    top5_sentiment_percent = sentiment_percent.loc[top5_sentiment_counts.index]

    traces = []
    for sentiment in top5_sentiment_counts.columns:
        trace = go.Bar(
            x=top5_sentiment_counts.index,
            y=top5_sentiment_counts[sentiment],
            name=sentiment,
            hoverinfo="x+y+text",
            text=top5_sentiment_percent[sentiment].round(1).astype(str) + "%",
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


def plot_top_organizations(tweet_df, top_n=5):
    """Affiche un graphique des principales organisations mentionnées"""
    df = tweet_df.explode("bert_orgs").dropna(subset=["bert_orgs"])
    value_counts = Counter(df["bert_orgs"])
    df_counts = pd.DataFrame(value_counts.items(), columns=["Value", "Count"]).sort_values(by="Count", ascending=False)

    df_top, fig = plot_top_orgs(df_counts, top_n=top_n)
    st.write(df_top)
    st.plotly_chart(fig)


def display_visualizations(tweet_df):
    """Gère l'affichage des visualisations"""
    visualize_option = st.sidebar.radio("Choose an option", ("Word Cloud", "Bar Plot"))

    if visualize_option == "Word Cloud":
        st.title("Word Cloud")
        generate_wordcloud(tweet_df["processed_text"])

    elif visualize_option == "Bar Plot":
        top_n = st.sidebar.slider("Number of top organizations to display", min_value=1, max_value=10, value=5)
        plot_top_organizations(tweet_df, top_n)


def display_predictions(tweet_df):
    """Affiche les prédictions et les graphiques associés"""
    tweet_df = process_tweet_data(tweet_df)

    fig = plot_pie_chart(tweet_df["polarity_predictions"])
    st.plotly_chart(fig)

    df = tweet_df.explode("bert_orgs").dropna(subset=["bert_orgs"])
    top_n = st.sidebar.slider("Number of top organizations to display", min_value=1, max_value=10, value=5)
    st.plotly_chart(plot_top_n_sentiment_multibar(df, top_n))
