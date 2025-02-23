# Twitter Stock Sentiment Analysis

## Description

Ce projet a pour objectif d'analyser les tweets relatifs aux marchés financiers, spécifiquement aux actions des entreprises technologiques. Le projet permet d'analyser les sentiments des utilisateurs Twitter à l'aide de modèles de traitement du langage naturel (NLP), comme **FinBERT**, afin de prédire les tendances du marché boursier à partir des discussions publiques sur Twitter.

### Fonctionnalités :
1. **Télécharger ou récupérer des tweets** : 
    - Télécharger un fichier CSV contenant déjà les tweets.
    - Utiliser l'API Twitter pour récupérer des tweets en temps réel via un `Bearer Token`.

2. **Analyse de données** :
    - Génération de **Word Clouds** pour visualiser les termes les plus fréquemment mentionnés.
    - Affichage de **Bar Plots** pour les entreprises les plus mentionnées sur Twitter.
    - Prédiction des tendances de sentiment du marché avec **FinBERT**.
    - Affichage de **Pie Charts** pour montrer la répartition des sentiments (positif, négatif, neutre).
    - Génération de **MultiBar Charts** pour visualiser les sentiments des principales entreprises discutées sur Twitter.

### Fonctionnalités supplémentaires à venir :
- Analyse des données boursières via **Yahoo Finance**.

## Prérequis

Avant d'exécuter ce projet, assurez-vous d'avoir installé les dépendances suivantes :

- **Python 3.x**
- **pip**

### Installer les dépendances :
```bash
pip install -r requirements.txt
