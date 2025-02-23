import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import plotly.express as px
from collections import Counter
from itertools import chain
import pandas as pd
import seaborn as sns

def generate_wordcloud(texts):
    """Créer et afficher un wordcloud avec Matplotlib"""
    wordcloud = WordCloud(max_words=100, width=800, height=400).generate(" ".join(texts))

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)


def plot_top_orgs(data, top_n=10):
    """
    Retourne un DataFrame des organisations les plus fréquentes et un graphique interactif.

    Paramètres :
    - data (pd.Series) : Série contenant des listes d'organisations.
    - top_n (int) : Nombre d'organisations à afficher (par défaut 10).

    Retour :
    - df_top_counts (pd.DataFrame) : DataFrame des organisations les plus fréquentes.
    - fig (plotly.graph_objects.Figure) : Graphique interactif Plotly.
    """
    all_values = list(chain(*data.apply(lambda x: list(set(x))).values))
    value_counts = Counter(all_values)

    df_counts = pd.DataFrame(value_counts.items(), columns=["Value", "Count"]).sort_values(by="Count", ascending=False)

    df_top_counts = df_counts.head(top_n)

    fig = px.bar(df_top_counts, x="Value", y="Count", title=f"Top {top_n} Organisations par Count",
                 labels={"Value": "Organisation", "Count": "Nombre d'occurrences"},
                 text_auto=True)

    return df_top_counts, fig


def plot_pie_chart(data):
    """
    Paramètres :
    - data (pd.Series) : Série de prédictions de polarité.

    Retour :
    - fig : Graphique circulaire interactif.
    """
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


import pandas as pd
import plotly.graph_objects as go


def plot_top5_sentiment_multibar(tweet_df):
    """
    Create an interactive multi-bar plot showing sentiment distribution for the top 5 companies
    with the highest number of positive sentiment tweets.

    Parameters:
    - tweet_df (pd.DataFrame): DataFrame with 'bert_orgs' (companies) and 'polarity_predictions' (sentiment categories).

    Returns:
    - fig (plotly.graph_objects.Figure): The interactive plotly figure.
    """

    df = tweet_df.explode("bert_orgs")

    sentiment_counts = df.groupby("bert_orgs")["polarity_predictions"].value_counts().unstack().fillna(0)

    sentiment_percent = sentiment_counts.div(sentiment_counts.sum(axis=1), axis=0) * 100

    top5_sentiment_counts = sentiment_counts.sort_values("positive", ascending=False).head(5)
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
        title="Top 5 Companies by Count of Positive Sentiment Tweets",
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
