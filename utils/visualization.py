import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import plotly.express as px
from collections import Counter
from itertools import chain
import pandas as pd

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
