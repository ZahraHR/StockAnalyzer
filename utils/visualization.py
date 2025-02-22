import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st


def generate_wordcloud(texts):
    """Créer et afficher un wordcloud avec Matplotlib"""
    wordcloud = WordCloud(max_words=100, width=800, height=400).generate(" ".join(texts))

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.pyplot(fig)
