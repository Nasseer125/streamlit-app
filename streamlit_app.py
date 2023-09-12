import pandas as pd
import streamlit as st
import joblib

# Chargement des données (contenant les noms des films, les identifiants de films et les genres)
data = pd.read_csv('votre_fichier_de_donnees.csv')

# Charger le modèle hybride
hybrid_model = joblib.load("svdpp_model.pkl")

# Interface utilisateur Streamlit
st.title("Système de recommandation de films")

# Champ de saisie pour l'identifiant ou le nom d'utilisateur
user_input = st.text_input('Entrez votre identifiant ou nom d\'utilisateur :')

if user_input:
    # Faites des recommandations avec le modèle
    user_id = trouver_id_utilisateur(user_input)  # Fonction pour trouver l'ID de l'utilisateur
    recommendations = faire_des_recommandations(model, user_id)

    # Correspondance des identifiants de films aux noms de films et aux genres
    movie_id_to_name = dict(zip(data['movieId'], data['title']))
    movie_id_to_genre = dict(zip(data['movieId'], data['genres']))

    # Affichage des recommandations de films avec les genres
    st.header('Films Recommandés avec Genres :')
    for movie_id in recommendations:
        st.write(f"Nom du Film : {movie_id_to_name[movie_id]}")
        st.write(f"Genres : {movie_id_to_genre[movie_id]}")
        st.write("\n")


def generate_recommendation(model, user_id, ratings_df, movies_df, n_items):
    # Obtenez une liste de tous les identifiants de films à partir de l'ensemble de données
    movie_ids = df_ratings["movieId"].unique()
    # Obtenir une liste de tous les films qui ont été regardés par l'utilisateur.
    movie_ids_user = df_ratings.loc[df_ratings["userId"] == user_id, "movieId"]
    # Obtenir une liste de tous les films IDS qui n'ont pas été regardés par l'utilisateur
    movie_ids_to_pred = np.setdiff1d(movie_ids, movie_ids_user)
 
   # Appliquer une note de 4 à toutes les interactions (uniquement pour correspondre au format de l'ensemble de données Surprise)
    test_set = [[user_id, movie_id, 4] for movie_id in movie_ids_to_pred]
 
   # Prévoir les évaluations et générer des recommandations
    predictions = model.test(test_set)
    pred_ratings = np.array([pred.est for pred in predictions])
    print("Top {0} recommandations d'éléments pour l'utilisateur {1}:".format(n_items, user_id))
   # Classer les films les plus populaires en fonction des évaluations prédites
    index_max = (-pred_ratings).argsort()[:n_items]
    for i in index_max:
        movie_id = movie_ids_to_pred[i]
        print(df_movies[df_movies["movieId"]==movie_id]["title"].values[0], pred_ratings[i])

