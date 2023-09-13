import pandas as pd
import streamlit as st
import joblib

# Charger le modèle hybride
hybrid_model = joblib.load("svdpp_model.pkl")

# Interface utilisateur Streamlit
st.title("Système de recommandation de films")

# Formulaire de saisie utilisateur
user_id = st.text_input("Entrez l'ID de l'utilisateur:")
movie_id = st.text_input("Entrez l'ID du film:")

if user_id and movie_id:
    try:
        user_id = int(user_id)
        movie_id = int(movie_id)

        # Faire une recommandation en utilisant le modèle hybride
        prediction = hybrid_model.estimate(user_id, movie_id)

        st.write(f"La prédiction de la note pour l'utilisateur {user_id} et le film {movie_id} est {prediction:.2f}")
    except ValueError:
        st.write("Veuillez entrer des valeurs numériques valides.")
